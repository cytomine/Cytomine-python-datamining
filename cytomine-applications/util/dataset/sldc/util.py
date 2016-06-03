# -*- coding: utf-8 -*-

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


def emplace(src, dest, mapping):
    """Place the values of src into dest at the indexes indicated by the mapping

    Parameters
    ----------
    src: iterable (size: n)
        Elements to emplace into the dest list
    dest: list (size: m)
        The list in which the elements of src must be placed
    mapping: iterable (subtype: int, size: n)
        The indexes of dest where the elements of src must be placed
    """
    for index, value in zip(mapping, src):
        dest[index] = value


def take(src, idx):
    """Generate a list containing the elements of src of which the index is contained in idx

    Parameters
    ----------
    src: list (size: n)
        Source iterable from which elements must be taken
    idx: iterable (subtype: int, range: [0, n[, size: m)
        Indexes iterable

    Returns
    -------
    list: list
        The list of taken elements
    """
    return [src[i] for i in idx]


def batch_split(n_batches, items):
    """Partition the items into a given number of batches of similar sizes (if the number of batches is greater than
    the number of items N in the topology, N batches are returned).

    Parameters
    ----------
    n_batches: int
        The number of batches
    items: iterable
        The elements to split into batches

    Returns
    -------
    batches: iterable (subtype: iterable (subtype: Tile), size: min(n_batches, N))
        The batches of tiles
    """
    item_count = len(items)
    if n_batches >= item_count:
        return [[item] for item in items]
    batches = [[] for _ in range(0, n_batches)]
    current_batch = 0
    bigger_batch_count = item_count % n_batches
    smaller_batch_size = (item_count / n_batches)
    bigger_batch_size = (item_count / n_batches) + 1
    for item in items:
        batches[current_batch].append(item)
        if (current_batch < bigger_batch_count and len(batches[current_batch]) >= bigger_batch_size) \
                or (current_batch >= bigger_batch_count and len(batches[current_batch]) >= smaller_batch_size):
            # check whether the current batch is full and should be changed
            current_batch += 1
    return batches
