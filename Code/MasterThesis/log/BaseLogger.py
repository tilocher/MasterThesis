import datetime

import torch
import wandb
import os
import atexit
import shutil

class LocalLogger():

    def __init__(self, name):


        self.ParentDirectory = name

        # Folder where this file is located
        self.Base_folder = os.path.dirname(os.path.realpath(__file__))

        # Create the main Folder for all runs of all instances
        if not 'runs' in os.listdir(self.Base_folder):
            os.makedirs(self.Base_folder + '\\runs')

        self.MakeBaseDirectory()


    def MakeBaseDirectory(self):
        # The full Folder name of the parent directory
        self.RunFolderName = self.Base_folder + '\\runs\\' + self.ParentDirectory

        # Calculate the next number for the filing system
        if not self.ParentDirectory in os.listdir(self.Base_folder + '\\runs'):
            os.makedirs(self.RunFolderName)

        # Filename with numbering system
        self.RunFileName = datetime.datetime.today().strftime('%d_%m___%H_%M')


    def AddLocalLogs(self, Logs:dict):

        self.Logs = Logs

        self.ManageFiles()


    def ManageFiles(self) -> None:

        # Create an additional folder for all other tracked objects if any
        if not ('Logs' in os.listdir(self.RunFolderName)) and any(self.Logs):
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

    def GetLocalSaveName(self, Log:str) -> str:

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


    def ForceClose(self) -> None:
        """
        Function to delete all files that have been created if an error occures.
        Added to prevent cluttering of files.
        :return:
        """

        # Delete all created files
        if self.RunFileName in os.listdir(self.RunFolderName): shutil.rmtree(self.RunFolderName + self.RunFileName)

        for FolderName in self.Logs.keys():

            if self.LogFileNames[FolderName] in os.listdir(self.LogFolderNames[FolderName]):
                os.remove(self.LogFolderNames[FolderName] + '\\' + self.LogFileNames[FolderName])



class Logger(LocalLogger):
    """
    Class to Create File structure and save Runs of pytorch models.
    """

    def __init__(self,name: str, debug:bool = False , BaseConfig = {}, AlreadyRunning = False) -> None:
        """
        Initialize Class member
        :param ParentDirectory: Name of the parent directory
        :param kwargs: additional structures to be saved, along with their file-extension
        """

        super(Logger, self).__init__(name)

        self.debug = debug

        self.BaseConfig = BaseConfig


        self.EntityName = 'tilocher-team'
        self.ProjectName = 'MasterThesis'

        if not debug and not AlreadyRunning:
            # Init WandB
            # self.api = wandb.Api()
            wandb.login()
            wandb.init(project=self.ProjectName,
                       name=self.RunFileName,
                       group=self.ParentDirectory,
                       config= BaseConfig)





    def ForceClose(self) -> None:
        """
        Function to delete all files that have been created if an error occures.
        Added to prevent cluttering of files.
        :return:
        """
        # Clear all local copies
        super().ForceClose()

        # if not self.debug:
        #     # Close the writer
        #     run = self.api.run(rf'{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}')
        #     run.delete()

    def SetParameter(self,config: dict):
        wandb.config.update(config)

    def GetConfig(self):
        return wandb.config






if __name__ == '__main__':

    a = Logger('first_test', debug= True)

