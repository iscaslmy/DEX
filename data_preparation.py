import samples_dao
import random

'''
从数据库读取数据, 对数据进行划分
'''


###########################################################
# 将所有的有标记数据划分为种子数据和待测试数据
# param training_percentage: [0,1]
def divide_samples_into_seeds_and_test(training_percentage=0.7, reset = True):
    if training_percentage < 0 or training_percentage > 1:
        raise ValueError('Wrong Parameter!')

    if not reset:
        return divide_stable_sample()

    # final results returned
    seeds = []
    test_data = []

    # read all labeled data
    all_labeled_samples = samples_dao.read_all_labeled_samples_by_story()

    # read all labeled story_ids
    story_ids = list(all_labeled_samples.keys())

    # calculate the size of seeds
    seed_size = int(len(story_ids) * training_percentage)

    seed_story_ids = []

    # build seeds
    for i in range(seed_size):
        # generate random integer to build seeds
        random_int = random.randint(0, len(story_ids) - 1)
        # all samples within story
        samples = all_labeled_samples[story_ids[random_int]]
        seed_story_ids.append(story_ids[random_int])
        # add samples into seeds
        seeds.extend(samples)
        # delete divided story_id
        del story_ids[random_int]

    # build test_data
    for story_id in story_ids:
        test_data.extend(all_labeled_samples[story_id])

    with open('../Archive/date_performance/seeds_story_ids.txt', 'w') as file:
        for line in seed_story_ids:
            file.write(line + '\n')

    null_seeds = []
    nonull_seeds = []
    # 对null进行抽样
    for seed in seeds:
        if len(seed.fps) == 1 and seed.fps[0] == 'null':
            null_seeds.append(seed)
        else:
            nonull_seeds.append(seed)

    # sampling_size = (int)(len(null_seeds)/3) if (int)(len(null_seeds)/3) <= len(null_seeds) else len(null_seeds)
    # temp_samples = random.sample(null_seeds, sampling_size)
    # nonull_seeds.extend(null_seeds)

    return nonull_seeds, test_data


def divide_stable_sample(seed_path='./Archive/date_performance/seeds_story_ids.txt'):
    with open(seed_path, 'r') as file:
        lines = file.readlines()

        # read all labeled data
        all_labeled_samples = samples_dao.read_all_labeled_samples_by_story()

        # final results returned
        seeds = []
        test_data = []

        # build seeds
        for i in lines:
            i = i.strip('\n')
            # all samples within story
            samples = all_labeled_samples[i]
            # add samples into seeds
            seeds.extend(samples)
            # delete divided story_id
            del all_labeled_samples[i]

        # build test_data
        for test_samples in all_labeled_samples.values():
            test_data.extend(test_samples)

        null_seeds = []
        nonull_seeds = []
        # 对null进行抽样
        for seed in seeds:
            if len(seed.fps) == 1 and seed.fps[0] == 'null':
                null_seeds.append(seed)
            else:
                nonull_seeds.append(seed)

        # sampling_size = (int)(len(null_seeds) / 3) if (int)(len(null_seeds) / 3) <= len(null_seeds) else len(null_seeds)
        # temp_samples = random.sample(null_seeds, sampling_size)
        # nonull_seeds.extend(null_seeds)

        return nonull_seeds, test_data


def divide_samples_into_10():
    '''
    从数据库中读取所有故事下的样本，划分成10折
    :return: 二维列表 10折和其中每一折的数据
    '''
    # final results
    results = []

    # all samples grouped by story_id
    all_samples = samples_dao.read_all_labeled_samples_by_story()

    # read all sentence from database
    all_story_ids = list(all_samples.keys())

    fold_size = int(len(all_story_ids) / 10)

    for i in range(9):
        per_fold = []

        for sample_index in range(fold_size):
            # generate sotry_id from all_story_id
            random_int = random.randint(0, len(all_story_ids) - 1)
            # all samples within story
            samples = all_samples[all_story_ids[random_int]]

            # iterate samples within stories
            for sample in samples:
                per_fold.append(sample)

            # remove story_id
            del all_story_ids[random_int]

        results.append(per_fold)

    # add last fold into results
    per_fold = []
    for sample_index in all_story_ids:
        samples = all_samples[sample_index]

        for sample in samples:
            per_fold.append(sample)

    results.append(per_fold)

    return results

