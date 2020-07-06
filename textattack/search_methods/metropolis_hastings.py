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
        self._lm_tokenizer = AutoTokenizer.from_pretrained(self.lm_type, use_fast=True)
        self._lm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
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
        input_ids = input_ids[: self._lm_tokenizer.model_max_length - 2]
        input_ids.insert(0, self._lm_tokenizer.bos_token_id)
        input_ids.append(self._lm_tokenizer.eos_token_id)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.to(self._lm_device)

        with torch.no_grad():
            loss = self._language_model(input_ids, labels=input_ids)[0].item()
            del input_ids

        perplexity = math.exp(loss)
        return 1 / perplexity

    def _batch_lm_score(self, text_list):
        """
        Assigns likelihood of a text as 1/perplexity(text) for list of texts in batch
        Args:
            text_list (list[str])
        Returns: 1/perplexity(text)
        """
        encoded_output = self._lm_tokenizer.batch_encode_plus(text_list)
        max_len = self._lm_tokenizer.model_max_length - 2
        input_ids = encoded_output["input_ids"]
        attention_mask = encoded_output["attention_mask"]
        for i in range(len(input_ids)):
            input_ids[i] = input_ids[i][:max_len]
            input_ids[i].insert(0, self._lm_tokenizer.bos_token_id)
            input_ids[i].append(self._lm_tokenizer.eos_token_id)
            # Append attention mask for BOS and EOS tokens
            attention_mask[i] += [1, 1]

        input_ids = utils.pad_lists(
            input_ids, pad_token=self._lm_tokenizer.pad_token_id
        )
        input_ids = torch.tensor(input_ids).to(self._lm_device)
        attention_mask = utils.pad_lists(attention_mask, pad_token=0)
        attention_mask = torch.tensor(attention_mask).to(self._lm_device)

        with torch.no_grad():
            logits = self._language_model(input_ids, attention_mask=attention_mask)[0]

        # Calculate loss for each text in batch
        losses = torch.zeros(input_ids.size()[0], dtype=torch.float)
        loss_func = torch.nn.CrossEntropyLoss(
            ignore_index=self._lm_tokenizer.pad_token_id
        )
        for i in range(losses.size()[0]):
            shift_logits = logits[i][:-1].contiguous()
            shift_labels = input_ids[i][1:].contiguous()
            losses[i] = loss_func(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            ).item()

        perplexities = torch.exp(losses)
        return (1 / perplexities).tolist()

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
        lm_scores = self._batch_lm_score([x.text for x in x_list])
        scores = [lm_scores[i] * goal_results[i].score for i in range(batch_size)]

        return scores, goal_results

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

                """
                According to Metropolis-Hastings algorithm
                let f(x) be value proportional to target distribution p(x)
                and g(x'|x) be transition probability from x to x'.
                Then, acceptance ratio = min(1, (f(x')*g(x|x')) / (f(x) * g(x'|x)))
                f(x) = LM(x) * C(y=t|x) and g(x'|x) is simply 1. 
                """
                candidates = list(range(len(transformations)))
                while candidates:
                    # Sample one candidate uniformly random
                    idx = np.random.choice(candidates)

                    acceptance_score = target_scores[idx] / population[k].target_score
                    acceptance_score = min(1, acceptance_score)
                    u = np.random.uniform(low=0.0, high=1.0)

                    if (
                        acceptance_score >= u
                        or goal_results[idx].goal_status
                        == GoalFunctionResultStatus.SUCCEEDED
                    ):
                        # Accept the proposed jump
                        population[k].result = goal_results[idx]
                        population[k].target_score = target_scores[idx]
                        break
                    else:
                        candidates.remove(idx)

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


class OldMetropolisHastingsSampling(SearchMethod):
    """ 
    Uses Metropolis-Hastings Sampling to generate adversarial samples.
    Based off paper "Generating Fluent Adversarial Examples for Natural Langauges" by Zhang, Zhou, Miao, Li (2019)
    Args:
        max_iter (int): The maximum number of sampling to perform. 
            If the word count of the text under attack is greater than `max_iter`, we replace max_iter with word count for that specific example.
            This is so we at least try to perturb every word once. 
        lm_type (str): The language model to use to estimate likelihood of text.
            Currently supported LM is "gpt-2"
    """

    def __init__(self, max_iter=200, lm_type="gpt2"):
        self.max_iter = max_iter
        self.lm_type = lm_type

        self._search_over = False
        self._lm_tokenizer = AutoTokenizer.from_pretrained(self.lm_type, use_fast=True)
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
        input_ids = input_ids[: self._lm_tokenizer.model_max_length - 2]
        input_ids.insert(0, self._lm_tokenizer.bos_token_id)
        input_ids.append(self._lm_tokenizer.eos_token_id)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.to(self._lm_device)

        with torch.no_grad():
            loss = self._language_model(input_ids, labels=input_ids)[0].item()
            del input_ids

        perplexity = math.exp(loss)
        return 1 / perplexity

    def _batch_lm_score(self, text_list):
        """
        Assigns likelihood of a text as 1/perplexity(text) for list of texts in batch
        Args:
            text_list (list[str])
        Returns: 1/perplexity(text)
        """
        encoded_output = self._lm_tokenizer.batch_encode_plus(text_list)
        max_len = self._lm_tokenizer.model_max_length - 2
        input_ids = encoded_output["input_ids"]
        attention_mask = encoded_output["attention_mask"]
        for i in range(len(input_ids)):
            input_ids[i] = input_ids[i][:max_len]
            input_ids[i].insert(0, self._lm_tokenizer.bos_token_id)
            input_ids[i].append(self._lm_tokenizer.eos_token_id)
            # Append attention mask for BOS and EOS tokens
            attention_mask[i] += [1, 1]

        input_ids = utils.pad_lists(
            input_ids, pad_token=self._lm_tokenizer.pad_token_id
        )
        input_ids = torch.tensor(input_ids).to(self._lm_device)
        attention_mask = utils.pad_lists(attention_mask, pad_token=0)
        attention_mask = torch.tensor(attention_mask).to(self._lm_device)

        with torch.no_grad():
            logits = self._language_model(input_ids, attention_mask=attention_mask)[0]

        # Calculate loss for each text in batch
        losses = torch.zeros(input_ids.size()[0], dtype=torch.float)
        loss_func = torch.nn.CrossEntropyLoss(
            ignore_index=self._lm_tokenizer.pad_token_id
        )
        for i in range(losses.size()[0]):
            shift_logits = logits[i][:-1].contiguous()
            shift_labels = input_ids[i][1:].contiguous()
            losses[i] = loss_func(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            ).item()

        perplexities = torch.exp(losses)
        return (1 / perplexities).tolist()

    def _batch_target_prob(self, x_list):
        """
        Calculates unnormalized estimation of target_prob(x) in batch
        target_prob(x) = lm_score(x) * prob_model(wrong|x)
        Args:
            attacked_text_list (list[AttackedText]): list of text we want to calculated probability of
        Returns: float representing target_prob(x)
        """
        batch_size = len(x_list)
        model_results, self._search_over = self.get_goal_results(x_list)
        lm_scores = self._batch_lm_score([x.text for x in x_list])
        scores = [lm_scores[i] * model_results[i].score for i in range(batch_size)]

        return scores, model_results

    def _normalize(self, values):
        """
        Take list of values and normalize it into a probability distribution
        """
        s = sum(values)
        if s == 0:
            return [1 / len(values) for v in values]
        else:
            return [v / s for v in values]

    def _perform_search(self, initial_result):
        num_words = len(initial_result.attacked_text.words)
        max_iter = max(self.max_iter, text_len)

        current_result = initial_result
        current_text = initial_result.attacked_text
        current_score = (
            self._lm_score(initial_result.attacked_text.text) * initial_result.score
        )

        for t in range(max_iter):
            # i-th word we want to transform
            i = n % num_words

            transformations = self.get_transformations(
                current_text,
                indices_to_modify=[i],
                original_text=initial_result.attacked_text,
            )

            if len(transformations) == 0:
                continue

            scores, model_results = self._batch_target_prob(transformations)
            if self._search_over:
                break
            proposal_dist = self._normalize(scores)
            # Choose one transformation randomly according to proposal distribution
            jump = np.random.choice(list(range(len(transformations))), p=proposal_dist)

            # Now we have calculate probability of return proposal g(x'|x)
            reverse_transformations = self.get_transformations(
                transformations[jump],
                indices_to_modify=[i],
                original_text=initial_result.attacked_text,
            )

            reverse_jump = None
            for k in range(len(reverse_transformations)):
                if reverse_transformations[k].text == current_text.text:
                    # Transition x -> x' exists
                    reverse_jump = k
                    break
            if not reverse_jump:
                return_prob = 0
            else:
                ret_scores, _ = self._batch_target_prob(reverse_transformations)
                return_prob = self._normalize(ret_scores)[reverse_jump]
                if self._search_over:
                    break

            """
            According to Metropolis-Hastings algorithm
            let f(x) be value proportional to target distribution p(x)
            and g(x|x') be transition probability from x' to x.
            Then, acceptance ratio = min(1, (f(x')*g(x|x')) / (f(x) * g(x'|x)))
            """
            acceptance_ratio = (scores[jump] * return_prob) / (
                current_score * proposal_dist[jump]
            )
            acceptance_ratio = min(1, acceptance_ratio)
            u = np.random.uniform(low=0.0, high=1.0)

            if (
                acceptance_ratio >= u
                or model_results[jump].goal_status == GoalFunctionResultStatus.SUCCEEDED
            ):
                # Accept the proposed jump
                current_result = model_results[jump]
                current_text = transformations[jump]
                current_score = scores[jump]

            if current_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                break

        return current_result
