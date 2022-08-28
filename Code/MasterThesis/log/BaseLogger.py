import datetime

import wandb
import os
import shutil
import yaml

# LOG_LOCATION = os.path.realpath(r"C:\Users\diloc\Desktop\Timur\Logs")
LOG_LOCATION = os.path.realpath(r"E:\MasterThesis\log")

class LocalLogger():

    log_location = LOG_LOCATION

    def __init__(self, name, BaseConfig={}):

        self.ParentDirectory = name

        # Folder where this file is located
        # self.Base_folder = os.path.relpath(os.path.dirname(os.path.realpath(__file__)), os.getcwd())
        if os.path.isdir(self.log_location):
            self.Base_folder = self.log_location
        else:
            self.Base_folder = os.path.relpath(os.path.dirname(os.path.realpath(__file__)), os.getcwd())

        self.Config = BaseConfig

        # Create the main Folder for all runs of all instances
        if not 'runs' in os.listdir(self.Base_folder):
            os.makedirs(self.Base_folder + '/runs')

        self.MakeBaseDirectory()

        self.ConfigFolderName = self.RunFolderName + '/Config'
        self.ConfigFileName =  f'Config_{self.RunFileName}.yaml'


        if not 'Config' in os.listdir(self.RunFolderName):
            os.makedirs(self.ConfigFolderName)

        self.SaveConfig(BaseConfig)



    def MakeBaseDirectory(self):
        # The full Folder name of the parent directory
        self.BaseRunFolderName = self.Base_folder + '/runs/' + self.ParentDirectory

        # Calculate the next number for the filing system
        if not self.ParentDirectory in os.listdir(self.Base_folder + '/runs'):
            os.makedirs(self.BaseRunFolderName)

        # Filename with numbering system
        self.RunFileName = datetime.datetime.today().strftime('%y_%m_%d___%H_%M')

        if not self.RunFileName in os.listdir(self.BaseRunFolderName):
            os.makedirs(self.BaseRunFolderName + '/' + self.RunFileName)

        self.RunFolderName = self.BaseRunFolderName + '/' + self.RunFileName

        # Create an additional folder for all other tracked objects if any
        if not ('Logs' in os.listdir(self.RunFolderName)):
            os.makedirs(self.RunFolderName + '/Logs')

    def SaveConfig(self, Config):

        self.Config.update(Config)

        with open(self.ConfigFolderName  + '/' + self.ConfigFileName, 'w') as file:
            yaml.dump(self.Config, file)
        return self.Config


    def AddLocalLogs(self, Logs: dict):

        self.Logs = Logs

        self.ManageFiles()

    def ManageFiles(self) -> None:

        # Init all additional file locations and names
        self.LogFolderNames = {}
        self.LogFileNames = {}

        # Create names and folder locations for additional objects
        for FolderName, FileExtension in self.Logs.items():

            FullFolderName = self.RunFolderName + '/Logs/' + FolderName

            FileName = f'{FolderName}' + FileExtension

            self.LogFolderNames.update({FolderName: FullFolderName})
            self.LogFileNames.update({FolderName: FileName})

            if not FolderName in os.listdir(self.RunFolderName + '/Logs'):
                os.makedirs(FullFolderName)


    def GetLocalSaveName(self, Log: str, prefix = '') -> str:

        """
        Get file locations of the objects that are tracked.
        :param Log: Name of the object to be saved
        :return: Full file location
        """
        if Log == 'run':
            ret = self.RunFolderName
        elif Log == 'Config':
            ret = self.ConfigFolderName + '/' + self.ConfigFileName
        else:
            ret = self.LogFolderNames[Log] + '/' + prefix + self.LogFileNames[Log]
        return ret

    def ForceClose(self) -> None:
        """
        Function to delete all files that have been created if an error occurs.
        Added to prevent cluttering of files.
        :return:
        """
        # pass
        # Delete all created files
        if self.RunFileName in os.listdir(self.BaseRunFolderName): shutil.rmtree(self.RunFolderName)
        print('Successfully removed all files')


    def GetConfig(self):

        return self.Config


class WandbLogger(LocalLogger):
    """
    Class to Create File structure and save Runs of pytorch models.
    """

    def __init__(self, name: str, group = '',BaseConfig={}) -> None:
        """
        Initialize Class member
        :param ParentDirectory: Name of the parent directory
        :param kwargs: additional structures to be saved, along with their file-extension
        """
        self.BaseConfig = BaseConfig

        self.EntityName = 'tilocher-team'
        self.ProjectName = 'MasterThesis'

        # Init WandB
        wandb.login()
        wandb.init(project=self.ProjectName,
                   name=name +'_'+ datetime.datetime.today().strftime('%d_%m___%H_%M'),
                   group=group,
                   config=BaseConfig)

        self.api = wandb.Api()



        super(WandbLogger, self).__init__(name)



    def ForceClose(self) -> None:
        """
        Function to delete all files that have been created if an error occures.
        Added to prevent cluttering of files.
        :return:
        """
        # Clear all local copies
        super().ForceClose()

        # Close the writer
        run = self.api.run(rf'{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}')
        run.delete()

    def SetParameter(self, config: dict):
        wandb.config.update(config)

    def GetConfig(self):
        return wandb.config

    def SaveConfig(self,Config):

        super().SaveConfig(Config)

        wandb.config.update(Config)

