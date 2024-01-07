from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


class ToxicDataset(Dataset):
    """
    A custom dataset class for handling toxic comments data.

    Args:
        tokens (list): List of tokens representing the comments.
        slicer (list): List of slicer values for the comments.
        labels (list): List of labels for the comments.

    Attributes:
        tokens (list): List of tokens representing the comments.
        slicer (list): List of slicer values for the comments.
        labels (list): List of labels for the comments.
    """

    def __init__(self, tokens, slicer, labels):
        """
        Initialize the ToxicDataset class.

        Args:
            tokens (list): List of tokens.
            slicer (object): Slicer object.
            labels (list): List of labels.
        """
        super(ToxicDataset, self).__init__()
        self.tokens = tokens
        self.labels = labels
        self.slicer = slicer

    def __len__(self):
        """
        Returns the length of the tokens list.

        Returns:
            int: The length of the tokens list.
        """
        return len(self.tokens)

    def __getitem__(self, index):
        """
        Retrieves the data at the given index.

        Args:
            index (int): The index of the data to retrieve.

        Returns:
            tuple: A tuple containing the tokens, slicer, and labels at the given index.
        """
        return self.tokens[index], self.slicer[index], self.labels[index]
    

class ToxicDataModule(pl.LightningDataModule):
    """
    LightningDataModule for handling toxic comment data.

    Args:
        cfg (object): Configuration object containing data-related settings.
        tokens (list): List of tokenized comments.
        slicer (object): Object used for slicing the tokenized comments.
        labels (list): List of labels corresponding to the tokenized comments.
    """

    def __init__(self, cfg, tokens, slicer, labels) -> None:
        """
        Initializes the ToxicDataModule class.

        Args:
            cfg (object): The configuration object.
            tokens (list): The list of tokens.
            slicer (object): The slicer object.
            labels (list): The list of labels.
        """
        super(ToxicDataModule, self).__init__()
        self.cfg = cfg
        self.tokens = tokens
        self.slicer = slicer
        self.labels = labels
        
    def setup(self, stage=None) -> None:
        """
        Setup method to prepare the train and validation datasets.

        Args:
            stage (str, optional): Stage of training. Defaults to None.
        """
        
        dataset = ToxicDataset(self.tokens, self.slicer, self.labels)
        self.train_set, self.val_set = torch.utils.data.random_split(
            dataset,
            [self.cfg.data.train_split, 1 - self.cfg.data.train_split]
        )
    
    def train_dataloader(self) -> DataLoader:
        """
        Returns the train DataLoader.

        Returns:
            DataLoader: Train DataLoader.
        """
        
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.cfg.data.batch_size,
            shuffle=self.cfg.data.shuffle,
        )
    
    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader: Validation DataLoader.
        """
        
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
        )