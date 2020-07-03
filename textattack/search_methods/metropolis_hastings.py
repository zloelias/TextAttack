import copy
import functools
import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import textattack
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared import utils


class PopulationMember:
    def __init__(self, result, target_score):
        self.result = result
        self.target_score = target_score
        self.alive = True


class MetropolisHastingsSampling(SearchMethod):
    """ 
    Uses Metropolis-Hastings Sampling to generate adversarial samples.
    Based off paper "Generating Fluent Adversarial Examples for Natural Langauges" by Zhang, Zhou, Miao, Li (2019)

    Args:
        max_iter (int): The maximum number of sampling to perform. 
            If the word count of the text under attack is greater than `max_iter`, we replace max_iter with word count for that specific example.
            This is so we at least try to perturb every word once. 
        lm_type (str): The language model to use to estimate likelihood of text.
            Currently supported LM is "gpt2"
    """

    def __init__(self, max_iter=50, pop_size=20, lm_type="gpt2"):
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.lm_type = lm_type

        self._search_over = False
        self._lm_tokenizer = AutoTokenizer.from_pretrained(self.lm_type)
        self._language_model = AutoModelForCausalLM.from_pretrained(self.lm_type)

        try:
            # Try to use GPU, but sometimes we might have out-of-memory issue
            # Having the below line prevents CUBLAS error when we try to switch to CPU
            torch.cuda.current_blas_handle()
            self._lm_device = utils.device
            self._language_model = self._language_model.to(self._lm_device)
        except RuntimeError as error:
            if "CUDA out of memory" in str(error):
                textattack.shared.utils.get_logger().warn(
                    "CUDA out of memory. Running GPT-2 for Metropolis Hastings on CPU."
                )
                self._lm_device = torch.device("cpu")
                self._language_model = self._language_model.to(self._lm_device)
            else:
                raise error

        self._language_model.eval()

    @functools.lru_cache(maxsize=2 ** 14)
    def _lm_score(self, text):
        """
        Assigns likelihood of a text as 1/perplexity(text)
        Args:
            text (str)
        Returns: 1/perplexity(text)
        """

        input_ids = self._lm_tokenizer.encode(text)
        input_ids = input_ids[:self._lm_tokenizer.model_max_length - 2]
        input_ids.insert(0, self._lm_tokenizer.bos_token_id)
        input_ids.append(self._lm_tokenizer.eos_token_id)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.to(self._lm_device)

        with torch.no_grad():
            loss = self._language_model(input_ids, labels=input_ids)[0].item()
            del input_ids

        perplexity = math.exp(loss)
        return 1 / perplexity

    def _batch_target_prob(self, x_list):
        """
        Calculates unnormalized estimation of target_prob(x) in batch
        target_prob(x) = lm_score(x) * prob_model(wrong|x)
        Args:
            attacked_text_list (list[AttackedText]): list of text we want to calculated probability of
        Returns: float representing target_prob(x)
        """
        batch_size = len(x_list)
        goal_results, self._search_over = self.get_goal_results(x_list)
        lm_scores = [self._lm_score(x.text) for x in x_list]
        scores = [lm_scores[i] * goal_results[i].score for i in range(batch_size)]

        return scores, goal_results

    def _normalize(self, values):
        """
        Take list of values and normalize it into a probability distribution
        """
        s = sum(values)
        if s == 0:
            return [1 / len(values) for v in values]
        else:
            return [v / s for v in values]

    def _calc_return_prob(self, proposed_state, previous_state, index, initial_result):
        # Now we have to calculate probability of return proposal g(x|x')
        reverse_transformations = self.get_transformations(
            proposed_state,
            indices_to_modify=[index],
            original_text=initial_result.attacked_text,
        )

        reverse_found = False
        for t in reverse_transformations:
            if t.text == previous_state.text:
                reverse_found == True
                break

        if reverse_found:
            return 1 / len(reverse_transformations)
        else:
            return 0.0

    def _perform_search(self, initial_result):
        text_len = len(initial_result.attacked_text.words)
        max_iter = max(self.max_iter, text_len)

        # target_score = LM(initial_text) * p(y_c | x_orig)
        initial_target_score = (
            self._lm_score(initial_result.attacked_text.text) * initial_result.score
        )
        initial_pop_member = PopulationMember(initial_result, initial_target_score)
        population = [copy.deepcopy(initial_pop_member) for _ in range(self.pop_size)]

        for _ in range(max_iter):
            for k in range(len(population)):
                modifiable_indices = (
                    set(range(text_len))
                    - population[k].result.attacked_text.attack_attrs[
                        "modified_indices"
                    ]
                )
                while modifiable_indices:
                    rand_idx = np.random.choice(tuple(modifiable_indices))
                    transformations = self.get_transformations(
                        population[k].result.attacked_text,
                        indices_to_modify=[rand_idx],
                        original_text=initial_result.attacked_text,
                    )

                    if len(transformations) > 0:
                        break
                    else:
                        modifiable_indices.remove(rand_idx)

                if not modifiable_indices:
                    # No transformations can be generated
                    # Kill this population member and repopulate
                    population[k].alive = False
                    continue

                target_scores, goal_results = self._batch_target_prob(transformations)

                if self._search_over:
                    break

                proposal_prob = 1 / len(transformations)
                # Select on transformation uniformly-random
                jump = np.random.choice(list(range(len(target_scores))))
                return_prob = self._calc_return_prob(
                    transformations[jump],
                    population[k].result.attacked_text,
                    rand_idx,
                    initial_result,
                )

                """
                According to Metropolis-Hastings algorithm
                let f(x) be value proportional to target distribution p(x)
                and g(x|x') be transition probability from x' to x.
                Then, acceptance ratio = min(1, (f(x')*g(x|x')) / (f(x) * g(x'|x)))
                """
                """
                acceptance_score = (target_scores[jump] * return_prob) / (
                    population[k].target_score * proposal_prob
                )
                """
                acceptance_score = target_scores[jump] / population[k].target_score
                acceptance_score = min(1, acceptance_score)
                u = np.random.uniform(low=0.0, high=1.0)
                if False:
                    print(f"target_score: {target_scores[jump]}")
                    print(f"return_prob: {return_prob}")
                    print(f"prev_score: {population[k].target_score}")
                    print(f"proposal_prob: {proposal_prob}")
                if (
                    acceptance_score >= u
                    or goal_results[jump].goal_status
                    == GoalFunctionResultStatus.SUCCEEDED
                ):
                    # Accept the proposed jump
                    population[k].result = goal_results[jump]
                    population[k].target_score = target_scores[jump]

            if self._search_over:
                break

            for k in range(len(population)):
                if not population[k].alive:
                    # Repopulate using original text
                    # TODO consider other way to repopulate?
                    population[k] = copy.deepcopy(initial_pop_member)

            population = sorted(population, key=lambda x: x.result.score, reverse=True)

            if (
                self._search_over
                or population[0].result.goal_status
                == GoalFunctionResultStatus.SUCCEEDED
            ):
                break

        return population[0].result

    def extra_repr_keys(self):
        return ["max_iter", "pop_size", "lm_type"]
