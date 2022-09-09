"""

    Template script for the submission. You can use this as a starting point for your code: you can
    copy this script as is into your repository, and then modify the associated Model class to include
    your logic, instead of the random baseline. Most of this script should be left unchanged for your submission
    as we should be able to run your code to confirm your scores.

    Please make sure you read and understand the competition rules and guidelines before you start.

"""

import os
from datetime import datetime
import yaml
import torch
import random
from dotenv import load_dotenv

# add parent folder as path
import sys
sys.path.insert(0, '../')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


# import env variables from file
load_dotenv('../upload.env', verbose=True)

# read yaml config
with open("../config.yml", "r") as file:
    config = yaml.safe_load(file)

set_seed(config['model_args']['seed'])

# variables for the submission
EMAIL = os.getenv('EMAIL')  # the e-mail you used to sign up
assert EMAIL != '' and EMAIL is not None
BUCKET_NAME = os.getenv('BUCKET_NAME')  # you received it in your e-mail
PARTICIPANT_ID = os.getenv('PARTICIPANT_ID')  # you received it in your e-mail
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')  # you received it in your e-mail
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')  # you received it in your e-mail


# run the evaluation loop when the script is called directly
if __name__ == '__main__':
    # import the basic classes
    from evaluation.EvalRSRunner import EvalRSRunner
    from evaluation.EvalRSRunner import ChallengeDataset
    from RecFormer.MyRunner import RecRunner
    print('\n\n==== Starting evaluation script at: {} ====\n'.format(datetime.utcnow()))
    # load the dataset
    print('\n\n==== Loading dataset at: {} ====\n'.format(datetime.utcnow()))
    # this will load the dataset with the default values for the challenge
    dataset = ChallengeDataset(num_folds=1, seed=42)
    print('\n\n==== Init runner at: {} ====\n'.format(datetime.utcnow()))
    # run the evaluation loop
    runner = EvalRSRunner(
        dataset=dataset,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        participant_id=PARTICIPANT_ID,
        bucket_name=BUCKET_NAME,
        email=EMAIL
        )
    print('==== Runner loaded, starting loop at: {} ====\n'.format(datetime.utcnow()))
    # NOTE: this evaluation will run with default values for the parameters and the upload flag
    # For local testing and iteration, you can check the tutorial in the notebooks folder and the
    # kaggle notebook: https://www.kaggle.com/code/vinidd/cikm-data-challenge
    my_model = RecRunner(
        tracks_info=dataset.df_tracks,
        user_info=dataset.df_users,
        # kwargs may contain additional arguments in case, for example, you 
        # have data augmentation functions that you wish to use in combination
        # with the dataset provided by the runner.
        config=config['model_args']
    )
    # run evaluation with your model
    # the evaluation loop will magically perform the fold splitting, training / testing
    # and then submit the results to the leaderboard
    runner.evaluate(
        model=my_model,
        upload=False
    )
    print('\n\n==== Evaluation ended at: {} ===='.format(datetime.utcnow()))