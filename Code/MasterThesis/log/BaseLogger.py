import datetime

import torch
import wandb
import os
import atexit
import shutil

class WB_Logger():
    """
    Class to Create File structure and save Runs of pytorch models.
    """

    def __init__(self,ParentDirectory: str, Logs: dict = {}, debug:bool = False) -> None:
        """
        Initialize Class member
        :param ParentDirectory: Name of the parent directory
        :param kwargs: additional structures to be saved, along with their file-extension
        """

        self.debug = debug

        # Parent directory file name
        self.ParentDirectory = ParentDirectory

        # Folder where this file is located
        self.Base_folder  = os.path.dirname(os.path.realpath(__file__))

        # Create the main Folder for all runs of all instances
        if not 'runs' in os.listdir(self.Base_folder):
            os.makedirs(self.Base_folder + '\\WandB_runs')

        # Additional  structures to be saved
        self.Logs = Logs

        # Assert that a dict is passed
        assert isinstance(self.Logs,dict), 'Logs must be a dict of Name, file-extension pairs'

        # Manage all the file structure and create folders, if they are missing
        self.ManageFiles()

        self.EntetyName = 'tilocher-team'
        self.ProjectName = 'MasterThesis'

        if not debug:
            # Init WandB
            self.api = wandb.Api()
            # wandb.login()
            # wandb.init(project=self.ProjectName,
            #            name=self.RunFileName,
            #            group=self.ParentDirectory)




    def ManageFiles(self) -> None:

        # The full Folder name of the parent directory
        self.RunFolderName = self.Base_folder + '\\WandB_runs\\' + self.ParentDirectory

        # Calculate the next number for the filing system
        if not self.ParentDirectory in os.listdir(self.Base_folder + '\\WandB_runs'):
            os.makedirs(self.RunFolderName)

        # Filename with numbering system
        self.RunFileName =  datetime.datetime.today().strftime('%d_%m___%H_%M')

        # Create an additional folder for all other tracked objects if any
        if not ('Logs' in os.listdir(self.RunFolderName)) and any(self.Logs) :
            os.makedirs(self.RunFolderName + '\\Logs')

        # Init all additional file locations and names
        self.LogFolderNames = {}
        self.LogFileNames = {}

        # Create names and folder locations for additional objects
        for FolderName, FileExtension in self.Logs.items():

            FullFolderName = self.RunFolderName +'\\Logs\\'+ FolderName

            FileName = f'{FolderName}_{self.RunFileName}' + FileExtension

            self.LogFolderNames.update({FolderName:FullFolderName})
            self.LogFileNames.update({FolderName: FileName})


            if not FolderName in os.listdir(self.RunFolderName + '\\Logs'):
                os.makedirs(FullFolderName)

        # Add \\ to foldername for later ease of use
        self.RunFolderName += '\\'



    def ForceClose(self) -> None:
        """
        Function to delete all files that have been created if an error occures.
        Added to prevent cluttering of files.
        :return:
        """
        # if not self.debug:
        #     # Close the writer
        #     run = self.api.run(rf'{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}')
        #     run.delete()

        # Delete all created files
        if self.RunFileName in os.listdir(self.RunFolderName): shutil.rmtree(self.RunFolderName + self.RunFileName)

        for FolderName in self.Logs.keys():

            if self.LogFileNames[FolderName] in os.listdir(self.LogFolderNames[FolderName]):
                os.remove(self.LogFolderNames[FolderName] + '\\' + self.LogFileNames[FolderName])


    def getSaveName(self, Log:str) -> str:
        """
        Get file locations of the objects that are tracked.
        :param Log: Name of the object to be saved
        :return: Full file location
        """
        if Log == 'run':
            ret = self.RunFolderName + self.RunFileName
        else:
            ret = self.LogFolderNames[Log] + '\\' + self.LogFileNames[Log]
        return ret



if __name__ == '__main__':

    a = WB_Logger('first_test')