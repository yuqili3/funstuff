import numpy as np
from heapq import heappush, heappop, heapify

'''
Suppose bernoulli random process X_i~Bern(p)
1. generate superletters of length n and correponding prob.
2. generate D-nary huffman code for each superletter
3. calculate the average code length for superletter and average code length for each letter
4. compare the entropy and average code length
'''

def generate_symb(p,n):
    N = 2**n # total number of symbols
    prob_symbol = []
    prob_dict = dict([])
    for i in range(N):
        bin_str = bin(i)[2:]
        ones = bin_str.count('1')
        prob_symbol.append([p**ones*(1-p)**(n-ones), [bin_str.zfill(n), ""]])
        prob_dict[bin_str.zfill(n)] = p**ones*(1-p)**(n-ones)
    return prob_symbol, prob_dict

#print(prob_symbol)
def encode_binary(prob_symbol):
    """Huffman encode the given dict mapping symbols to weights"""
#    heap = [[prob, [sym, ""]] for sym, prob in symbol_prob.item()]
    heap = prob_symbol.copy()
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0]+hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key = lambda p: (len(p[-1]),p))

def encode_Dnary(prob_symbol,D):
    """Huffman encode the given dict mapping symbols to weights"""
#    heap = [[prob, [sym, ""]] for sym, prob in symbol_prob.item()]
    heap = prob_symbol.copy()
    heapify(heap)
    while len(heap) > 1:
        entries = []
        prob_sum = 0
        codes = []
        for i in range(D):
            if len(heap) >= 1:
                entries.append(heappop(heap))
                for pair in entries[i][1:]:
                    pair[1] = str(i)+pair[1]
                prob_sum += entries[i][0]
                codes += entries[i][1:]
        heappush(heap,  [prob_sum] + codes)
    return sorted(heappop(heap)[1:], key = lambda p: (len(p[-1]),p))

def get_avg_len(huff,n):
    L_S = 0  # Average length of code of superletter
    for symb in huff:
        L_S += len(symb[1]) * prob_dict[symb[0]]
    return L_S/n

if __name__ == '__main__':
    # example of Bern(0.8), D-nary huffman code
    p = 0.8
    n = 3
    D = 2
    prob_symbol, prob_dict = generate_symb(p, n)
    #huff = encode_binary(prob_symbol)
    huff = encode_Dnary(prob_symbol,D)
    print (f'Symbol\tProb\tHuffman Code')
    for symb in huff:
        print("%s\t%.4f\t%s" % (symb[0], prob_dict[symb[0]], symb[1]))
    
    HX = -p*np.log2(p)-(1-p)*np.log2(1-p) # entropy
    #HX=0
    print('H(x): %.4f' %(HX))
    L_S = get_avg_len(huff, n)
    print('average huffman code length of superletter (len=%d): %.4f' %(n,L_S*n))
    print('average huffman code length per letter: %.4f' %(L_S))
    print('redundancy: %.4f' %(L_S-HX))
    
    for n in np.arange(1,15):
        prob_symbol, prob_dict = generate_symb(p, n)
        huff = encode_Dnary(prob_symbol, D)
        L_S = get_avg_len(huff, n)
        print('n = %d : avg len = %.4f' %(n,L_S))