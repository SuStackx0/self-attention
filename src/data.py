"""Character-level dataset using a tiny Shakespeare-like text. No internet connection needed."""

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Embedded training text — no file I/O, no internet needed
# ---------------------------------------------------------------------------

TINY_TEXT = """
HAMLET: To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause.

OPHELIA: My lord, how does your honour for this many a day?

HAMLET: I humbly thank you; well, well, well.

OPHELIA: My lord, I have remembrances of yours
That I have longed long to re-deliver.
I pray you now receive them.

HAMLET: No, not I, I never gave you aught.

KING: Though yet of Hamlet our dear brother's death
The memory be green, and that it us befitted
To bear our hearts in grief, and our whole kingdom
To be contracted in one brow of woe,
Yet so far hath discretion fought with nature
That we with wisest sorrow think on him
Together with remembrance of ourselves.
Therefore our sometime sister, now our queen,
The imperial jointress to this warlike state,
Have we, as 'twere with a defeated joy,
With an auspicious and a dropping eye,
With mirth in funeral and with dirge in marriage,
In equal scale weighing delight and dole,
Taken to wife.

POLONIUS: My liege and madam, to expostulate
What majesty should be, what duty is,
What day is day, night night, and time is time,
Were nothing but to waste night, day, and time.
Therefore, since brevity is the soul of wit,
And tediousness the limbs and outward flourishes,
I will be brief. Your noble son is mad.
Mad call I it; for, to define true madness,
What is't but to be nothing else but mad?
But let that go.

QUEEN: More matter with less art.

POLONIUS: Madam, I swear I use no art at all.
That he is mad, 'tis true; 'tis true 'tis pity,
And pity 'tis 'tis true—a foolish figure,
But farewell it, for I will use no art.
Mad let us grant him then; and now remains
That we find out the cause of this effect—
Or rather say the cause of this defect,
For this effect defective comes by cause.
Thus it remains, and the remainder thus.

HORATIO: My lord, I came to see your father's funeral.

HAMLET: I prithee do not mock me, fellow student.
I think it was to see my mother's wedding.

HORATIO: Indeed, my lord, it followed hard upon.

HAMLET: Thrift, thrift, Horatio! The funeral baked meats
Did coldly furnish forth the marriage tables.
Would I had met my dearest foe in heaven
Or ever I had seen that day, Horatio!

MARCELLUS: Something is rotten in the state of Denmark.

HORATIO: Heaven will direct it.

GHOST: I am thy father's spirit,
Doom'd for a certain term to walk the night,
And for the day confined to fast in fires,
Till the foul crimes done in my days of nature
Are burnt and purg'd away. But that I am forbid
To tell the secrets of my prison-house,
I could a tale unfold whose lightest word
Would harrow up thy soul, freeze thy young blood,
Make thy two eyes like stars start from their spheres,
Thy knotted and combined locks to part,
And each particular hair to stand on end
Like quills upon the fretful porpentine.
But this eternal blazon must not be
To ears of flesh and blood.

HAMLET: O all you host of heaven! O earth! What else?
And shall I couple hell? O fie! Hold, hold, my heart,
And you, my sinews, grow not instant old,
But bear me stiffly up. Remember thee?
Ay, thou poor ghost, whiles memory holds a seat
In this distracted globe. Remember thee?
Yea, from the table of my memory
I'll wipe away all trivial fond records,
All saws of books, all forms, all pressures past
That youth and observation copied there,
And thy commandment all alone shall live
Within the book and volume of my brain,
Unmix'd with baser matter. Yes, by heaven!

LAERTES: And so have I a noble father lost,
A sister driven into desperate terms,
Whose worth, if praises may go back again,
Stood challenger on mount of all the age
For her perfections. But my revenge will come.

KING: Break not your sleeps for that. You must not think
That we are made of stuff so flat and dull
That we can let our beard be shook with danger
And think it pastime. You shortly shall hear more.
I lov'd your father, and we love ourself,
And that, I hope, will teach you to imagine—

FORTINBRAS: Go, captain, from me greet the Danish king;
Tell him that by his licence Fortinbras
Craves the conveyance of a promis'd march
Over his kingdom. You know the rendezvous.
If that his majesty would aught with us,
We shall express our duty in his eye,
And let him know so.
"""


# ---------------------------------------------------------------------------
# Vocabulary utilities
# ---------------------------------------------------------------------------

def build_vocab(text: str):
    """
    Build a character-level vocabulary from a text string.

    Args:
        text: the full training text

    Returns:
        chars: sorted list of unique characters
        stoi:  dict mapping character → integer index
        itos:  dict mapping integer index → character
    """
    # Sort for determinism: same text always produces the same mapping
    chars = sorted(set(text))

    stoi = {ch: i for i, ch in enumerate(chars)}  # string to index
    itos = {i: ch for i, ch in enumerate(chars)}   # index to string

    return chars, stoi, itos


def encode(text: str, stoi: dict) -> list:
    """
    Convert a string to a list of integer token ids.

    Args:
        text: input string
        stoi: character-to-index mapping from build_vocab

    Returns:
        list of integers, same length as text
    """
    return [stoi[ch] for ch in text]


def decode(ids: list, itos: dict) -> str:
    """
    Convert a list of integer token ids back to a string.

    Args:
        ids:  list of integers (or a 1-D tensor)
        itos: index-to-character mapping from build_vocab

    Returns:
        decoded string
    """
    # Handle both plain lists and PyTorch tensors
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    return ''.join(itos[i] for i in ids)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Sliding-window character-level language modelling dataset.

    For each index i, returns a pair:
        input_ids:  data[i : i + seq_len]       — the context
        target_ids: data[i+1 : i + seq_len + 1] — the next character at each step

    Example (seq_len=5, data="Hello world"):
        i=0 → input="Hello", target="ello "
        i=1 → input="ello ", target="llo w"
        ...

    The model learns: given the context, predict each next character.
    """

    def __init__(self, data: list, seq_len: int):
        """
        Args:
            data:    list of integer token ids (the encoded text)
            seq_len: number of tokens in each training window
        """
        self.data    = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        # Each window needs seq_len tokens for input plus 1 more for the last target.
        # So the last valid start index is len(data) - seq_len - 1.
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int):
        """
        Returns a training pair at position idx.

        Returns:
            input_ids:  LongTensor of shape (seq_len,)
            target_ids: LongTensor of shape (seq_len,) — shifted one step ahead
        """
        # Slice a window of seq_len tokens starting at idx
        chunk = self.data[idx : idx + self.seq_len + 1]  # length seq_len + 1

        input_ids  = torch.tensor(chunk[:-1], dtype=torch.long)  # (seq_len,)
        target_ids = torch.tensor(chunk[1:],  dtype=torch.long)  # (seq_len,)

        return input_ids, target_ids


# ---------------------------------------------------------------------------
# Convenience function: build everything from raw text
# ---------------------------------------------------------------------------

def get_dataloaders(text: str, seq_len: int, batch_size: int, split: float = 0.9):
    """
    Build train and validation DataLoaders from a raw text string.

    Steps:
        1. Build vocabulary
        2. Encode text to integer ids
        3. Split into train (first `split` fraction) and val (remainder)
        4. Wrap each split in a CharDataset and DataLoader

    Args:
        text:       full training text (e.g. TINY_TEXT)
        seq_len:    context window length
        batch_size: number of sequences per batch
        split:      fraction of data used for training (default 0.9 = 90%)

    Returns:
        train_loader: DataLoader for training
        val_loader:   DataLoader for validation
        vocab:        tuple of (chars, stoi, itos)
    """
    # Step 1: Build vocabulary
    chars, stoi, itos = build_vocab(text)

    # Step 2: Encode the entire text into a flat list of token ids
    data = encode(text, stoi)  # list of ints, length = len(text)

    # Step 3: Split
    n_train = int(len(data) * split)
    train_data = data[:n_train]
    val_data   = data[n_train:]

    # Step 4: Create Dataset objects
    train_dataset = CharDataset(train_data, seq_len)
    val_dataset   = CharDataset(val_data,   seq_len)

    # Step 5: Wrap in DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,         # shuffle training data each epoch
        drop_last=True,       # discard the last incomplete batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,        # keep val order fixed for reproducibility
        drop_last=True,
    )

    return train_loader, val_loader, (chars, stoi, itos)
