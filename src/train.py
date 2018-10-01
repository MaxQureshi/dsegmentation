import time
from netutil.SegmentNet import SegmentNet
from netutil.DyMapNet import DyMapNet

def main():
    sortingDL = SegmentNet('defaults')
    # sortingDL = DyMapNet('defaults')
    start=time.time()
    sortingDL.do_train()
    print('Total Training Time {:.3f}'.format(time.time()-start))

if __name__ == '__main__':
    main()
