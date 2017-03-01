---
author: leon
comments: true
date: 2015-09-15 16:40:05+00:00
layout: post
title: '[leetcode] Leetcode(5)-Manacher算法解决最大回文串问题' 
categories:
- leetcode
- 算法

tags:
- leetcode
- 算法
- cpp

---


在leetcode上算法题目第5号题目会遇到最大回文串问题，暴力计算复杂度高达O(n^2)不可取，可使用经典的Manacher算法解决。

### 问题

输入字符串S，求S包含的最长回文串

### 预处理

Manacher用一个巧妙的方式，将所有可能的奇数/偶数长度的回文子串都转换成了奇数长度：在每个字符的两边都插入一个特殊的符号。如 `xxyy` => `#x#x#y#y#`,`yxy` => `#y#x#y#`

### 记录数组

记录数组P[]用以记录以当前S[i]为中心的回文串向左/右方向上的最大扩展长度。比如S和P的对应关系：

```
S # 1 # 2 # 2 # 1 # 2 # 3 # 2 # 1 #
P 0 1 0 1 4 1 0 3 0 1 0 5 0 1 0 1 0
```
这里有一个重要的特性：`P[i]正好是原字符串S中回文串的总长度`，因此利用这个特性，计算完P[]就可以得知最大的回文串。

### 记录数组计算过程的优化

主要思想是：利用回文串的对称性，`计算中心点右边子回文串时可以利用左边对称位置已经计算出的值`。

先增加两个辅助变量id和mx，其中id表示最大回文子串中心的位置，mx则为id+P[id]，也就是最大回文子串的边界。j=2*id-i，即j是i关于id的对称点， 那么会出现一下两种情况：

(1) 以j为中心的回文串没有超过mx的边界，此时P[i]=P[j]

(2) 以j为中心的回文串超过mx的边界，此时[i,mx]范围内的串已经比较过了属于P[i]回文串的一部分，可以直接从max开始接着计算P[i]超出mx的那部分回文串。
经过上面两步处理，用id、mx配合，可以在每次循环中直接对P[i]的快速赋值，从而在计算以i为中心的回文子串的过程中，不必每次都从1开始比较，减少了比较次数，最终使得求解最长回文子串的长度达到线性O(N)的时间复杂度。

### 实现

我的提交代码：

```cpp
#include <string>
#include <iterator>
#include <iostream>
#include <vector>

using namespace std;

class LongestPalindromeSolution {
public:
    string longestPalindrome(string sStr){
        string pStr("#");
        string res;
        int pLen = convert(sStr,pStr);
        vector<int> table(pLen);

        int center = 2;
        int scopeL;
        int scopeR;
        int offset = 1;

        int tmpL;
        int tmpR;
        int tmpRMax;
        int centerMax;

        int max = 0;
        int maxPos = 0;

        table[0] = 0;
        table[1] = 1;
        maxPos = 1;
        max = 1;

        while(center < pLen){
            scopeL = center -offset;
            scopeR = center +offset;

            // pidx
            while(scopeL >= 0 && scopeR < pLen){
                if(pStr.at(scopeL) == pStr.at(scopeR)){
                    scopeL--;
                    scopeR++;
                    table[center]++;
                }else{
                    break;
                }
            }

            centerMax = center + table[center];
            if( table[center] > max ){
                maxPos = center;
                max = table[center];
            }

            // pidx+?
            for(tmpR = center+1; tmpR < scopeR; ){
                tmpL = 2*center-tmpR;
                tmpRMax = table[tmpL] + tmpR;
                if( tmpRMax < centerMax ){
                    table[tmpR] = table[tmpL];
                    tmpR++;
                }else{
                    table[tmpR] = centerMax - tmpR;
                    offset = table[tmpR]+1;
                    break;
                }
            }

            // reset pIdx
            center = tmpR;
            printVector(table);
        }

        res = sStr.substr(((maxPos-table[maxPos])>>1),table[maxPos]);
        cout << maxPos << ":" << table[maxPos] << ":" << res << endl;
        return res;
    }

    int convert(string &src, string& dst){
        int index;
        int len = src.size();

        cout << "src:#-";
        for(index = 0; index < len; index++){
            dst.push_back(src.at(index));
            cout << src.at(index) << " ";
            dst.append("#");
            cout << "#"  << " ";
        }
        cout << endl;
        return dst.size();
    }
private:
    void printVector(vector<int> &vec){
        cout << "res:";
        for(unsigned int i = 0; i < vec.size()-1; i++){
            cout << vec.at(i) << " ";
        }
        cout << endl;
        cout.flush();
    }
};

int main()
{
    string dst("#");
    string src("1232454232414211");
    string src2("12784487sad22s2f");
    string src3("a");
    string src4("b2");

    LongestPalindromeSolution solu;
    cout << solu.longestPalindrome(src3) << endl;
    cout << solu.longestPalindrome(src4) << endl;

    //cout << solu.convert(src,dst) << ": " << dst;
    return 0;
}

```

最终测试运行耗时8ms，运行效率超过了72%的答案哈哈。
