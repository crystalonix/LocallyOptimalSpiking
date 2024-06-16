#
# Copyright (c) 2024 Anik Chattopadhyay, Arunava Banerjee
#
# Author: Anik Chattopadhyay
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/
#
# Note: This project is also subject to a provisional patent. The Creative Commons license
# applies to the documentation and code provided herein, but does not grant any rights to
# the patented invention.
#
import file_utils


def encrypt_wav_file_into_spike_format(filename):
    file_utils.read_wav_file_into_array(filename)
