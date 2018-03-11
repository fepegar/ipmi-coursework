#!/usr/bin/env python3

import ipmi


def main():
    unsegmented_subjects = ipmi.get_unsegmented_subjects()
    for subject in unsegmented_subjects:
        print('Segmenting', subject)
        subject.segment()


if __name__ == '__main__':
    main()
