import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import os
import atexit
import shutil

class TensorBoard_BaseLogger():
    """
    Class to Create File structure and save Runs of pytorch models.
    """

    def __init__(self,ParentDirectory: str, Logs: dict = {}) -> None:
        """
        Initialize Class member
        :param ParentDirectory: Name of the parent directory
        :param kwargs: additional structures to be saved, along with their file-extension
        """

        # Parent directory file name
        self.ParentDirectory = ParentDirectory

        # Folder where this file is located
        self.Base_folder  = os.path.dirname(os.path.realpath(__file__))

        # Create the main Folder for all runs of all instances
        if not 'runs' in os.listdir(self.Base_folder):
            os.makedirs(self.Base_folder+ '\\runs')

        # Additional  structures to be saved
        self.Logs = Logs

        # Assert that a dict is passed
        assert isinstance(self.Logs,dict), 'Logs must be a dict of Name, file-extension pairs'

        # Manage all the file structure and create folders, if they are missing
        self.ManageFiles()

        # Init TensorBoard writer
        self.writer = SummaryWriter(self.RunFolderName + self.RunFileName)

        # Launch a local TensorBoard server
        self.LaunchTensorBoard()

        # Register a function at interpreter exit, to delay the shutdown of the TensorBoard server
        atexit.register(self.EndOfScript)


    def ManageFiles(self) -> None:

        # The full Folder name of the parent directory
        self.RunFolderName = self.Base_folder + '\\runs\\' + self.ParentDirectory

        # Calculate the next number for the filing system
        if not self.ParentDirectory in os.listdir(self.Base_folder + '\\runs'):
            os.makedirs(self.RunFolderName)
            self.run_number = 0
        else:
            all_runs = list(filter(lambda x: 'run_' in x,os.listdir(self.RunFolderName)))

            if len(all_runs) > 0:
                all_runs_sorted = sorted(all_runs, key=lambda x: int(x[4:]))
                self.run_number = int(all_runs_sorted[-1][4:]) + 1
            else:
                self.run_number = 0

        # Filename with numbering system
        self.RunFileName = 'run_{}'.format(self.run_number)

        # Create an additional folder for all other tracked objects if any
        if not ('Logs' in os.listdir(self.RunFolderName)) and any(self.Logs) :
            os.makedirs(self.RunFolderName + '\\Logs')

        # Init all additional file locations and names
        self.LogFolderNames = {}
        self.LogFileNames = {}

        # Create names and folder locations for additional objects
        for FolderName, FileExtension in self.Logs.items():

            FullFolderName = self.RunFolderName +'\\Logs\\'+ FolderName

            FileName = f'{FolderName}_run_{self.run_number}' + FileExtension

            self.LogFolderNames.update({FolderName:FullFolderName})
            self.LogFileNames.update({FolderName: FileName})


            if not FolderName in os.listdir(self.RunFolderName + '\\Logs'):
                os.makedirs(FullFolderName)

        # Add \\ to foldername for later ease of use
        self.RunFolderName += '\\'



    def EndOfScript(self) -> None:
        """
        Function to delay closure of local TensorBoard server
        :return:
        """
        self.writer.flush()
        print('\n\n')
        input('---Script completed press "Enter" twice to close TensorBoard server---')

    def LaunchTensorBoard(self) -> None:
        """
        Launch a local TensorBoard Server
        :return:
        """
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.RunFolderName])
        url = tb.launch()
        print(f'Tensorboard listening on {url}')

    def ForceClose(self) -> None:
        """
        Function to delete all files that have been created if an error occures.
        Added to prevent cluttering of files.
        :return:
        """
        # Shutdown TensorBoard on error
        atexit.unregister(self.EndOfScript)

        # Close the writer
        self.writer.close()

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

