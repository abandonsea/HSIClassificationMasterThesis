#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 17:39 2021

@author: Pedro Vieira
@description: Implements the train function for the 3DCRN network
"""

import shutil
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.crn3d import CRN3D
from test import test_model
from utils.config import CRN3DConfig
from utils.tools import *
from utils.dataset import CRN3DDataset

# Import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Train
def train():
    cfg = CRN3DConfig('config.yaml')

    # Start tensorboard
    writer = None
    if cfg.use_tensorboard:
        writer = SummaryWriter(cfg.tensorboard_folder)

    # Load raw dataset, apply PCA and normalize dataset.
    data = HSIData(cfg.dataset, cfg.data_folder, cfg.sample_bands)

    # Load a checkpoint
    if cfg.use_checkpoint:
        print('Loading checkpoint')
        value_states, train_states, best_model_state = load_checkpoint(cfg.checkpoint_folder,
                                                                       cfg.checkpoint_file)
        first_run, first_epoch, loss_state, correct_state = value_states
        model_state, optimizer_state, scheduler_state = train_states
        best_model, best_accuracy = best_model_state
        if first_epoch == cfg.num_epochs - 1:
            first_epoch = 0
            first_run += 1
        print(f'Loaded checkpoint with run {first_run} and epoch {first_epoch}')
    else:
        first_run, first_epoch, loss_state, correct_state = (0, 0, 0.0, 0)
        model_state, optimizer_state, scheduler_state = None, None, None
        best_model, best_accuracy = None, 0

        # Save data for tests if we are not loading a checkpoint
        data.save_data(cfg.exec_folder)

    # Run training
    print(f'Starting experiment with {cfg.num_runs} run' + ('s' if cfg.num_runs > 1 else ''))
    for run in range(first_run, cfg.num_runs):
        print(f'STARTING RUN {run + 1}/{cfg.num_runs}')

        # Generate samples or read existing samples
        if cfg.generate_samples and first_epoch == 0:
            train_gt, test_gt, val_gt = data.sample_dataset(cfg.train_split, cfg.val_split, cfg.max_samples)
            HSIData.save_samples(train_gt, test_gt, val_gt, cfg.split_folder, cfg.train_split, cfg.val_split, run)
        else:
            train_gt, _, val_gt = HSIData.load_samples(cfg.split_folder, cfg.train_split, cfg.val_split, run)

        # Select most valuable samples
        print(f'SELECTING VALUABLE SAMPLES {run + 1}/{cfg.num_runs}')
        train_gt = select_valuable_samples(data.image, train_gt, cfg.svm_num_select, cfg.svm_train_fraction,
                                           cfg.svm_init_sample_size)

        # Create train and test dataset objects
        train_dataset = CRN3DDataset(data.image, train_gt, cfg.sample_size, data_augmentation=True)
        val_dataset = CRN3DDataset(data.image, val_gt, cfg.sample_size, data_augmentation=False)

        # Create train and test loaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=cfg.test_batch_size, shuffle=False)

        # Setup model, optimizer, loss and scheduler
        assert cfg.sample_size == 23, "Samples must have size 23! Change architecture for another window size"
        model = nn.DataParallel(CRN3D(cfg.sample_bands, data.num_classes))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step, gamma=cfg.gamma)

        # Start counting loss and correct predictions
        running_loss = 0.0
        running_correct = 0

        # Load variable states when loading a checkpoint
        if cfg.use_checkpoint:
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            lr_scheduler.load_state_dict(scheduler_state)
            running_loss = loss_state
            running_correct = correct_state

        # Enable GPU training
        model = model.to(device)
        criterion = criterion.to(device)

        # Save best models per run
        run_best_model = None
        run_best_accuracy = 0.0

        # Run epochs
        total_steps = len(train_loader)
        for epoch in range(first_epoch, cfg.num_epochs):
            print("STARTING EPOCH {}/{}".format(epoch + 1, cfg.num_epochs))

            # Run iterations
            for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
                # images should have size [batch_size, 1, 10, 13, 13]
                images = images.unsqueeze(1).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                # Compute running loss and number of correct predictions for printing
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                running_correct += (predicted == labels).sum().item() / labels.shape[0]

                # Print steps and loss every PRINT_FREQUENCY
                if (i + 1) % cfg.print_frequency == 0:
                    avg_loss = running_loss / cfg.print_frequency
                    accuracy = running_correct / cfg.print_frequency
                    running_loss = 0.0
                    running_correct = 0

                    # Print data
                    tqdm.write(
                        f'\tEpoch [{epoch + 1}/{cfg.num_epochs}] Step [{i + 1}/{total_steps}]'
                        f'\tLoss: {avg_loss:.5f}\tAccuracy: {accuracy:.5f}')

                    # Compute intermediate results for visualization
                    if writer is not None:
                        writer.add_scalar('training loss', avg_loss, epoch * total_steps + i)
                        writer.add_scalar('accuracy', accuracy, epoch * total_steps + i)

            # Reset running loss and correct
            running_loss = 0.0
            running_correct = 0

            # Run validation
            if cfg.val_split > 0:
                print("STARTING VALIDATION {}/{}".format(epoch + 1, cfg.num_epochs))
                model.eval()
                report = test_model(model, val_loader)
                model.train()

                # Save validation results
                filename = cfg.results_folder + 'validations.txt'
                save_results(filename, report, run, epoch, validation=True)

                accuracy = report['overall_accuracy']
                if accuracy > run_best_accuracy:
                    run_best_accuracy = accuracy
                    run_best_model = model.state_dict()

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model.state_dict()

            # Save checkpoint
            checkpoint = {
                'run': run,
                'epoch': epoch,
                'loss_state': running_loss,
                'correct_state': running_correct,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': lr_scheduler.state_dict(),
                'best_accuracy': best_accuracy,
                'best_model': best_model
            }
            torch.save(checkpoint,
                       cfg.checkpoint_folder + 'checkpoint_run_' + str(run) + '_epoch_' + str(epoch) + '.pth')

        # Reset first epoch in case a checkpoint was loaded
        first_epoch = 0

        # Save trained model
        model_file = cfg.exec_folder + f'runs/crn3d_model_run_{run}.pth'
        run_best_file = cfg.exec_folder + f'runs/crn3d_best_model_run_{run}.pth'
        torch.save(model.state_dict(), model_file)
        torch.save(run_best_model, run_best_file)
        print(f'Finished training run {run + 1}')

    # Save the best model
    best_model_file = cfg.exec_folder + 'best_model.pth'
    torch.save(best_model, best_model_file)

    if cfg.delete_checkpoints:
        shutil.rmtree(cfg.checkpoint_folder)

    if cfg.use_tensorboard:
        writer.close()


# Main function
def main():
    train()


if __name__ == '__main__':
    main()
