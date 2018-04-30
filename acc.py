with open('pred.txt') as pred, open('../data/testY.txt') as real:
        cnt, total = 0, 0
        for l1, l2 in zip(pred, real):
                cnt += sum(p == r for p, r in zip(l1, l2))
                total += len(l1)
        print(100. * cnt / total)
