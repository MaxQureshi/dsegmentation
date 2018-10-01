import time
from netutil.SegmentNet import SegmentNet

def main():
    sortingDL = SegmentNet('defaults')
    start=time.time()
    sortingDL.do_train()
    print('Total Training Time {:.3f}'.format(time.time()-start))

if __name__ == '__main__':
    main()
