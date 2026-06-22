"""Library for reading in TMSi files and converting to MNE RawArray objects.

The following module is a modification of a few classes from a larger code base
created by Twente Medical Systems International B.V., Oldenzaal The
Netherlands. Some docstrings, formatting, variables, variable names and even
classes have been changed.

The Twente Medical Systems International B.V.
lisencing information is as below:

(c) 2022 Twente Medical Systems International B.V.,
Oldenzaal The Netherlands

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import logging
import struct
import typing
from pathlib import Path

import mne
import numpy as np

logger = logging.getLogger(__name__)

POLY5_VERSION_NUMBER = 203


class Poly5Reader:
    """Reader for TMSi Poly5 files.

    This class allows reading in various file types
    created on TMSi devices and/or in Poly5 format.

    """

    def __init__(
        self,
        filename: str | Path | None = None,
        read_all: bool = True,
        verbose: bool = True,
    ):
        if filename is None:
            import tkinter as tk  # noqa: PLC0415
            from tkinter import filedialog  # noqa: PLC0415

            root = tk.Tk()
            filename = filedialog.askopenfilename()
            root.withdraw()

        self.filename: Path = Path(filename) if filename is not None else Path()
        self.read_all: bool = read_all
        self.verbose: bool = verbose
        if self.verbose:
            logger.info("Reading file %s", self.filename)
        self._readFile(self.filename)

    def read_data_MNE(  # noqa: N802
        self,
    ) -> mne.io.RawArray:
        """Return MNE RawArray given internal channel names and types.

        Returns:
            mne.io.RawArray: The raw data as an MNE RawArray object.
        """
        streams = self.channels
        fs = self.sample_rate
        labels = [s.name for s in streams]
        units = [s.unit_name for s in streams]

        type_options = [
            "ecg",
            "bio",
            "stim",
            "eog",
            "misc",
            "seeg",
            "dbs",
            "ecog",
            "mag",
            "eeg",
            "ref_meg",
            "grad",
            "emg",
            "hbr",
            "hbo",
        ]
        types_clean: list[str] = []
        for t in labels:
            for t_option in type_options:
                if t_option in t.lower():
                    types_clean.append(t_option)
                    break
            else:
                types_clean.append("misc")

        info = mne.create_info(ch_names=labels, sfreq=fs, ch_types=str(types_clean))

        # convert from microvolts to volts if necessary
        scale = np.array([1e-6 if u == "µVolt" else 1 for u in units])

        return mne.io.RawArray(self.samples * np.expand_dims(scale, axis=1), info)

    def _readFile(self, filename: Path) -> None:  # noqa: N802
        # """number_per_block = self.num_samples_per_block
        # suko = (self.num_samples % number_per_block) * self.num_channels"""
        self.file_obj = filename.open("rb")
        file_obj = self.file_obj
        self._readHeader(file_obj)
        self.channels = self._readSignalDescription(file_obj)
        number_per_block = self.num_samples_per_block
        self._myfmt = "f" * self.num_channels * number_per_block
        self._buffer_size = self.num_channels * number_per_block

        if self.read_all:
            sample_buffer = np.zeros(self.num_channels * self.num_samples)

            for i in range(self.num_data_blocks):
                # """numb_per_block = self.num_data_blocks
                # print("\rProgress: % 0.1f %%"
                #  %(100*i/numb_per_block), end="\r")"""

                # Check whether final data block is
                # filled completely or not
                if i == self.num_data_blocks - 1:
                    d_smp_bk = self.num_samples / self.num_data_blocks
                    _final_block_size = d_smp_bk
                    number_per_block = self.num_samples_per_block
                    if _final_block_size % number_per_block != 0:
                        numb_blk = self.num_samples_per_block
                        n_samps = self.num_samples
                        n_chan = self.num_channels
                        suko = (n_samps % numb_blk) * n_chan
                        data_block = self._readSignalBlock(
                            file_obj,
                            buffer_size=suko,
                            myfmt="f" * suko,
                        )
                    else:
                        data_block = self._readSignalBlock(
                            file_obj,
                            self._buffer_size,
                            self._myfmt,
                        )
                else:
                    data_block = self._readSignalBlock(
                        file_obj,
                        self._buffer_size,
                        self._myfmt,
                    )

                # Get indices that need to be
                # filled in the samples array
                number_per_block = self.num_samples_per_block
                i1 = i * number_per_block * self.num_channels
                i2 = (i + 1) * number_per_block * self.num_channels

                # Correct for final data block if
                # this is not fully filled
                i2 = min(self.num_samples * self.num_channels, i2)

                # Insert the read data_block into
                # the sample_buffer array
                sample_buffer[i1:i2] = data_block

            samples = np.transpose(
                np.reshape(
                    sample_buffer,
                    [self.num_samples, self.num_channels],
                )
            )

            self.ch_names = [s.name for s in self.channels]
            self.ch_unit_names = [s.unit_name for s in self.channels]
            self.samples = samples
            if self.verbose:
                logger.info("Done reading data.")
            self.file_obj.close()

    def readSamples(self, n_blocks: int | None = None) -> np.ndarray:  # noqa: N802
        """Function to read a subset of sample blocks from a file."""
        if n_blocks is None:
            n_blocks = self.num_data_blocks

        sample_buffer = np.zeros(
            self.num_channels * n_blocks * self.num_samples_per_block
        )

        for i in range(n_blocks):
            data_block = self._readSignalBlock(
                self.file_obj,
                self._buffer_size,
                self._myfmt,
            )
            i1 = i * self.num_samples_per_block * self.num_channels
            i2 = (i + 1) * self.num_samples_per_block * self.num_channels
            sample_buffer[i1:i2] = data_block

        return np.transpose(
            np.reshape(
                sample_buffer,
                [self.num_samples_per_block * (n_blocks), self.num_channels],
            )
        )

    def _readHeader(self, f: typing.BinaryIO) -> None:  # noqa: N802
        header_data = struct.unpack(
            "=31sH81phhBHi4xHHHHHHHiHHH64x",
            f.read(217),
        )
        magic_number = str(header_data[0])
        version_number = header_data[1]
        self.sample_rate: int = header_data[3]
        # """self.storage_rate=header_data[4]
        self.num_channels: int = header_data[6] // 2
        self.num_samples: int = header_data[7]
        # """self.start_time = datetime.datetime(header_data[8], header_data[9],
        #                                     header_data[10], header_data[12],""""
        #                                     header_data[13], header_data[14])""""
        self.num_data_blocks: int = header_data[15]
        self.num_samples_per_block: int = header_data[16]
        if magic_number != "b'POLY SAMPLE FILEversion 2.03\\r\\n\\x1a'":
            logger.error("This is not a Poly5 file.")
        elif version_number != POLY5_VERSION_NUMBER:
            logger.error("Version number of file is invalid.")
        elif self.verbose:
            msg = f"\t Number of samples:  {self.num_samples} "
            logger.info(msg)
            msg = f"\t Number of channels:  {self.num_channels} "
            logger.info(msg)
            msg = f"\t Sample rate: {self.sample_rate} Hz"
            logger.info(msg)

    def _readSignalDescription(self, f: typing.BinaryIO) -> list[Channel]:  # noqa: N802
        chan_list = []
        for _ in range(self.num_channels):
            channel_description = struct.unpack("=41p4x11pffffH62x", f.read(136))
            name = channel_description[0][5:].decode("ascii")
            unit_name = channel_description[1].decode("utf-8")
            chan_list.append(Channel(name, unit_name))
            f.read(136)
        return chan_list

    def _readSignalBlock(  # noqa: N802
        self, f: typing.BinaryIO, buffer_size: int, myfmt: str
    ) -> np.ndarray:
        f.read(86)
        sampledata = f.read(buffer_size * 4)
        datablock = struct.unpack(myfmt, sampledata)
        return np.asarray(datablock)

    def close(self) -> None:
        """Function to close off file object after reading."""
        self.file_obj.close()


class Channel:
    """'Channel' represents a device channel.

    Attributes:
        name (str): The name of the channel.
        unit_name (str): The name of unit (e.g. "μVolt") of channel
            sample-data.
    """

    @property
    def name(self) -> str:
        """Return the name of the channel."""
        return self.__name

    @property
    def unit_name(self) -> str:
        """Return the unit name of the channel."""
        return self.__unit_name

    def __init__(self, name: str, unit_name: str):
        self.__unit_name = unit_name
        self.__name = name


if __name__ == "__main__":
    data = Poly5Reader()
