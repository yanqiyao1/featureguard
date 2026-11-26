"""This helper is shared by training and serving; its argument decides the order."""


def pack(frame, columns):
    return frame.loc[:, columns].to_numpy()
