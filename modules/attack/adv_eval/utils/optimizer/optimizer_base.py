from loguru import logger


class BaseOptimizer:
    def __init__(
        self,
        task,
        gold_evaluator,
        victim_evaluator,
        attack_instance
    ):
        self.task = task
        self.gold_evaluator = gold_evaluator
        self.victim_evaluator = victim_evaluator
        self.attack_instance = attack_instance
        self.negative_optimization_goal = True
        if self.negative_optimization_goal:
            logger.info('use negative optimization goal')
        self.meta_prompt = None
        # self.num_iter = 30
        self.num_iter = 10
        self.early_stop_threshold = 80
        self.num_best_sample = 10
        # self.num_sample_per_iter = 8
        self.num_sample_per_iter = 4

    def get_name(self):
        raise NotImplementedError

    def combine_score(self, gold_score, victim_score):
        if self.negative_optimization_goal:
            return round(victim_score - gold_score, 1)
        else:
            return round(gold_score - victim_score, 1)

    def evaluate(self):
        dataset = self.task
        # dataset = self.task.dataset
        rtn = {
            'context': [],
            'init_response': [],
            'adv_response': [],
            'init_gold_score': [],
            'adv_gold_score': [],
            'init_victim_score': [],
            'adv_victim_score': [],
        }

        for item in dataset:
            logger.info(f'len(item):{len(item)}')
            if (len(item) == 4) or (len(item) == 6):
                # for JudgeLab datasets
                
                context = item['source']
                init_response = item['target']
                ground_truth = init_response
                logger.info(f'ground_truth:{ground_truth}')
                context2 = item
            else:
                raise NotImplementedError
            perturbed_response = self.perturb(context, init_response, ground_truth, context2)
            # logger.info(f'perturbed_response:{perturbed_response}')
            init_gold_score = self.gold_evaluator.get_score(context, init_response, ground_truth, context2)
            adv_gold_score = self.gold_evaluator.get_score(context, perturbed_response, ground_truth, context2)
            init_victim_score = self.victim_evaluator.get_score(context, init_response, ground_truth, context2)
            adv_victim_score = self.victim_evaluator.get_score(context, perturbed_response, ground_truth, context2)
            logger.info(
                f'gold score: {init_gold_score} -> {adv_gold_score}'
            )
            logger.info(
                f'victim score: {init_victim_score} -> {adv_victim_score}'
            )
            logger.info('********************************************')

            rtn['context'].append(context)
            rtn['init_response'].append(init_response)
            rtn['adv_response'].append(perturbed_response)
            rtn['init_gold_score'].append(init_gold_score)
            rtn['adv_gold_score'].append(adv_gold_score)
            rtn['init_victim_score'].append(init_victim_score)
            rtn['adv_victim_score'].append(adv_victim_score)

        return rtn

    def get_query_prompt(self, context, response_text_score, context2=None):
        raise NotImplementedError

    def fetch_new_responses(self, query_prompt, attack_instance):
        raise NotImplementedError

    def perturb(self, context, init_response, ground_truth, context2=None):
        # logger.info(f'context:{context}\ninit_response:{init_response}\nground_truth:{ground_truth}\nperturb--context2:{context2}')
        try:
            gold_score = self.gold_evaluator.get_score(context, init_response, ground_truth, context2)
        except Exception as e:
            logger.info(f'Exception:{str(e)}')
        # logger.info(f'gold_score:{gold_score}')
        victim_score = self.victim_evaluator.get_score(context, init_response, ground_truth, context2)
        # logger.info(f'victim_score:{victim_score}')
        response_list = [[init_response, self.combine_score(gold_score, victim_score)]]
        response_set = set()
        response_set.add(init_response)
        # logger.info(f'response_set:{response_set}')

        for _ in range(self.num_iter):
            response_list = sorted(response_list, key=lambda x: x[1], reverse=False)
            # > 90 then early stop
            if response_list[-1][1] > self.early_stop_threshold:
                return response_list[-1][0]

            response_text_score = ''
            if self.num_best_sample != -1:
                if len(response_list) > self.num_best_sample:
                    current_response_list = response_list[-self.num_best_sample:]
                else:
                    current_response_list = response_list
            else:
                current_response_list = response_list

            for item in current_response_list:
                response, score = item[0], item[1]
                response_text_score += f'text:\n{response}\nscore:\n{score}\n'

            response_text_score = response_text_score.strip()
            # logger.info(f'response_text_score:{response_text_score}')
            query_prompt = self.get_query_prompt(context, response_text_score, context2)
            # logger.info(f'query_prompt:{query_prompt}')
            new_responses = self.fetch_new_responses(query_prompt, self.attack_instance)
            # logger.info(f'new_responses:{new_responses}')
            if len(new_responses) == 0:
                break

            for new_response in new_responses:
                # deduplicate
                if new_response in response_set:
                    continue
                else:
                    response_set.add(new_response)

                gold_score = self.gold_evaluator.get_score(context, new_response, ground_truth, context2)
                victim_score = self.victim_evaluator.get_score(context, new_response, ground_truth, context2)
                # gold score = -1 refers to unsafe response
                if gold_score == -1:
                    continue
                if victim_score == -1:
                    continue
                response_list.append([new_response, self.combine_score(gold_score, victim_score)])
        response_list = sorted(response_list, key=lambda x: x[1], reverse=False)
        logger.info(f'response_list={response_list}')
        logger.info(f'final query samples={len(response_list)}')
        best_response = response_list[-1][0]
        return best_response
