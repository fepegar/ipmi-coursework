#!/usr/bin/env python3

import ipmi


def main():
    subjects = ipmi.get_segmented_subjects()
    subjects.extend(ipmi.get_unsegmented_subjects())
    for subject in subjects:
        print('Segmenting', subject)
        subject.segment()


if __name__ == '__main__':
    main()
