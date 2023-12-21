import matplotlib.pyplot as plt
import numpy as np
import os


def make_representation_1():
    fig, ax = plt.subplots(figsize=(15, 5))
    # Training and evaluation period
    plt.plot(range(1, 13), np.ones(12), marker='o', color='g', linestyle='--', label='Training period')
    plt.plot(range(13, 25), np.ones(12), marker='o', color='b', linestyle='--', label='Evaluation period')
    for i in range(1, 5):
        plt.plot([x + i for x in range(1, 13)], np.ones(12) + i, marker='o', color='g', linestyle='--')
        plt.plot([x + i for x in range(13, 25)], np.ones(12) + i, marker='o', color='b', linestyle='--')

    # Prediction period
    i += 1
    plt.plot([x + i for x in range(1, 13)], np.ones(12) + i, marker='o', color='g', linestyle='--')
    plt.plot([x + i for x in range(13, 25)], np.ones(12) + i, marker='o', color='r', linestyle='--',
             label='Prediction period')

    plt.xlabel('Month Id')
    plt.ylabel('Batch Id')
    plt.grid()
    plt.legend(loc='lower left', bbox_to_anchor=(0.75, 1.0))

    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'ml_model_senario1'))


def make_representation_2():
    fig, ax = plt.subplots(figsize=(15,5))
    plt.plot(range(1,13),np.ones(12), marker='o', color='g',linestyle='--', label='Training period')
    plt.plot(range(13,16),np.ones(3), marker='o', color='b',linestyle='--', label='Evaluation period')

    for i in range(1,5):
        plt.plot([x+i for x in range(1,13)],np.ones(12)+i, marker='o', color='g',linestyle='--')
        plt.plot([x+i for x in range(13,16)],np.ones(3)+i, marker='o', color='b',linestyle='--')

    # prediction period
    i+=1
    plt.plot([x+i for x in range(1,13)],np.ones(12)+i, marker='o', color='g',linestyle='--')
    plt.plot([x+i for x in range(13,16)], np.ones(3)+i, marker='o', color='r', linestyle='--', label='Prediction period')
    plt.xlabel('Month Id')
    plt.ylabel('Batch Id')
    plt.grid()
    plt.legend(loc='lower left', bbox_to_anchor=(0.75, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join('plots','ml_model_senario2'))


def make_representation_3a():
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.plot(range(1, 13), np.ones(12), marker='o', color='g', linestyle='--', label='Training period')
    plt.plot(range(13, 16), np.ones(3), marker='o', color='b', linestyle='--', label='Evaluation period')
    for i in range(1, 5):
        plt.plot([x + i for x in range(1, 13)], np.ones(12) + i, marker='o', color='g', linestyle='--')
        plt.plot([x + i for x in range(13, 16)], np.ones(3) + i, marker='o', color='b', linestyle='--')

    plt.xlabel('Month Id')
    plt.ylabel('Batch Id')
    plt.grid()
    plt.legend(loc='lower left', bbox_to_anchor=(0.75, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'ml_model_senario_3a'))


def make_representation_3b():
    fig, ax = plt.subplots(figsize=(15,5))
    plt.plot(range(1,13),np.ones(12), marker='o', color='g',linestyle='--', label='Training period')
    plt.plot(range(13,16),np.ones(3), marker='o', color='gray',linestyle='--', label='Not used')
    plt.plot(range(16,19),np.ones(3), marker='o', color='b',linestyle='--', label='Evaluation period')

    for i in range(1,5):
        plt.plot([x+i for x in range(1,13)],np.ones(12)+i, marker='o', color='g',linestyle='--')
        plt.plot([x+i for x in range(13,16)],np.ones(3)+i, marker='o', color='gray',linestyle='--')
        plt.plot([x+i for x in range(16,19)],np.ones(3)+i, marker='o', color='b',linestyle='--')

    plt.xlabel('Month Id')
    plt.ylabel('Batch Id')
    plt.grid()
    plt.legend(loc='lower left', bbox_to_anchor=(0.75, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join('plots','ml_model_senario_3b'))


def make_representation_3c():
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.plot(range(1, 13), np.ones(12), marker='o', color='g', linestyle='--', label='Training period')
    plt.plot(range(13, 19), np.ones(6), marker='o', color='gray', linestyle='--', label='Not used')
    plt.plot(range(19, 22), np.ones(3), marker='o', color='b', linestyle='--', label='Evaluation period')

    for i in range(1, 5):
        plt.plot([x + i for x in range(1, 13)], np.ones(12) + i, marker='o', color='g', linestyle='--')
        plt.plot([x + i for x in range(13, 19)], np.ones(6) + i, marker='o', color='gray', linestyle='--')
        plt.plot([x + i for x in range(19, 22)], np.ones(3) + i, marker='o', color='b', linestyle='--')

    plt.xlabel('Month Id')
    plt.ylabel('Batch Id')
    plt.grid()
    plt.legend(loc='lower left', bbox_to_anchor=(0.75, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'ml_model_senario_3c'))


def make_representation_3d():
    fig, ax = plt.subplots(figsize=(15,5))
    plt.plot(range(1,13),np.ones(12), marker='o', color='g',linestyle='--', label='Training period')
    plt.plot(range(13,22),np.ones(9), marker='o', color='gray',linestyle='--', label='Not used')
    plt.plot(range(22,25),np.ones(3), marker='o', color='b',linestyle='--', label='Evaluation period')

    for i in range(1,5):
        plt.plot([x+i for x in range(1,13)],np.ones(12)+i, marker='o', color='g',linestyle='--')
        plt.plot([x+i for x in range(13,22)],np.ones(9)+i, marker='o', color='gray',linestyle='--')
        plt.plot([x+i for x in range(22,25)],np.ones(3)+i, marker='o', color='b',linestyle='--')

    plt.xlabel('Month Id')
    plt.ylabel('Batch Id')
    plt.grid()
    plt.legend(loc='lower left', bbox_to_anchor=(0.75, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join('plots','ml_model_senario_3d'))


def make_representation_4():
    fig, ax = plt.subplots(figsize=(15, 5))
    # Training and evaluation period
    plt.plot(range(1, 13), np.ones(12), marker='o', color='g', linestyle='--', label='Training period')
    plt.plot(range(13, 25), np.ones(12), marker='o', color='b', linestyle='--', label='Evaluation period')
    for i in range(1, 5):
        plt.plot([x + i for x in range(1, 13)], np.ones(12) + i, marker='o', color='g', linestyle='--')
        plt.plot([x + i for x in range(13, 25)], np.ones(12) + i, marker='o', color='b', linestyle='--')

    # Prediction period
    i += 1
    plt.plot([x + i for x in range(1, 13)], np.ones(12) + i, marker='o', color='g', linestyle='--')
    plt.plot([x + i for x in range(13, 25)], np.ones(12) + i, marker='o', color='r', linestyle='--',
             label='Prediction period')

    plt.xlabel('Month Id')
    plt.ylabel('Batch Id')
    plt.grid()
    plt.legend(loc='lower left', bbox_to_anchor=(0.75, 1.0))

    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'ml_model_senario_4'))


def main():
    ''' Is a stand alone script that is used to generate the diagrams/figures that explain how the train/test/prediction
    is done in terms of what data intervals are used, to build the models. Each dot shows a particular month of
    sales_batch, the color represents if it is part of the training data, evaluation_data / prediction .'''

    # scenario 1: 12 month training, 12 testing, 12 month predicting
    make_representation_1()

    # scenario 2: 12 month training, next 1 month testing, 1 month predicting
    make_representation_2()

    # scenario 3A: 12 month training, testing 12+1,12+2,12+3
    make_representation_3a()

    # scenario 3B: 12 month training,  step 3months testing month 12+4,12+5,12+6
    make_representation_3b()

    # scenario 3C: 12 month training,  step 3months testing month 12+7,12+8,12+9
    make_representation_3c()


    # scenario 3D: 12 month training,  step 3months testing month 12+10,12+11,12+12
    make_representation_3d()

    # scenario 4: 12 month training, 12 testing, 12 month predicting
    make_representation_4()


if __name__ == '__main__':
  to_create_paths = ['plots']
  for my_path in to_create_paths:
    if not os.path.exists(my_path):
      # Create the directory
      os.makedirs(my_path)

    main()

