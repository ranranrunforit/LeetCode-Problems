# Day 1 Prefix Sum
# 1480. Running Sum of 1d Array
'''
Prefix Sum
Explanation
Let B[i] = A[0] + A[1] + .. + A[i]
B[i] = B[i-1] + A[i]


Complexity
Time O(N)
Space O(N)
Space O(1) if changing the input, like in Java.
'''
class Solution(object):
    def runningSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        return nums
# Python 3 version
class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        return list(itertools.accumulate(nums))
# 724. Find Pivot Index
# DP
class Solution(object):
    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l, r = 0, sum(nums)
        for i in range(len(nums)):
            r -= nums[i]
            if l == r: return i
            l += nums[i]
        return -1
'''
Approach #: Prefix Sum
Complexity Analysis

Time Complexity: O(N), where NN is the length of nums.
Space Complexity: O(1), the space used by leftsum and S.
'''
class Solution(object):
    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        S = sum(nums)
        leftsum = 0
        for i, x in enumerate(nums):
            if leftsum == (S - leftsum - x):
                return i
            leftsum += x
        return -1
# Day 2 String
# 205. Isomorphic Strings
class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        s2t, t2s = {}, {}
        for i in range(len(s)):
            if s[i] in s2t and s2t[s[i]] != t[i]:
                return False
            if t[i] in t2s and t2s[t[i]] != s[i]:
                return False
            s2t[s[i]] = t[i]
            t2s[t[i]] = s[i]
        return True

class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        return len(set(s)) == len(set(t)) == len(set(zip(s, t)))

# 392. Is Subsequence

# Day 3 Linked List
# 21. Merge Two Sorted Lists

# 206. Reverse Linked List

# Day 4 Linked List
# 876. Middle of the Linked List

# 142. Linked List Cycle II

# Day 5 Greedy
# 121. Best Time to Buy and Sell Stock

# 409. Longest Palindrome

# Day 6 Tree
# 589. N-ary Tree Preorder Traversal

# 102. Binary Tree Level Order Traversal

# Day 7 Binary Search
# 704. Binary Search

# 278. First Bad Version

# Day 8 Binary Search Tree
# 98. Validate Binary Search Tree

# 235. Lowest Common Ancestor of a Binary Search Tree

# Day 9 Graph/BFS/DFS
# 733. Flood Fill

# 200. Number of Islands

# Day 10 Dynamic Programming
# 509. Fibonacci Number

# 70. Climbing Stairs

# Day 11 Dynamic Programming
# 746. Min Cost Climbing Stairs

# 62. Unique Paths

# Day 12 Sliding Window/Two Pointer
# 438. Find All Anagrams in a String

# 424. Longest Repeating Character Replacement
''' Sliding Window, just O(n)
Explanation
maxf means the max frequency of the same character in the sliding window.
To better understand the solution,
you can firstly replace maxf with max(count.values()),
Now I improve from O(26n) to O(n) using a just variable maxf.


Complexity
Time O(n)
Space O(128)
'''
class Solution(object):
    def characterReplacement(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        maxf = res = 0
        count = collections.Counter()
        for i in range(len(s)):
            count[s[i]] += 1
            maxf = max(maxf, count[s[i]])
            if res - maxf < k:
                res += 1
            else:
                count[s[i - res]] -= 1
        return res
'''
Solution 2
Another version of same idea.
In a more standard format of sliding window.
Maybe easier to understand

Time O(N)
Space O(26)'''
class Solution(object):
    def characterReplacement(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        maxf = i = 0
        count = collections.Counter()
        for j in range(len(s)):
            count[s[j]] += 1
            maxf = max(maxf, count[s[j]])
            if j - i + 1 > maxf + k:
                count[s[i]] -= 1
                i += 1
        return len(s) - i

# Day 13 Hashmap
# 1. Two Sum

# 299. Bulls and Cows
'''
Let us first evaluate number of bulls B: 
by definition it is number of places with the same digit in secret and guess: 
so let us just traverse our strings and count it.
Now, let us evaluate both number of cows and bulls: B_C: 
we need to count each digit in secret and in guess and 
choose the smallest of these two numbers. Evaluate sum for each digit.
Finally, number of cows will be B_C - B, so we just return return the answer!
Complexity: both time and space complexity is O(1). 
Imagine, that we have not 4 lengths, but n, 
then we have O(n) time complexity and O(10) space complexity to keep our counters.
'''
class Solution(object):
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """
        B = sum([x==y for x,y in zip(secret, guess)])
        Count_sec = Counter(secret)
        Count_gue = Counter(guess)
        B_C = sum([min(Count_sec[elem], Count_gue[elem]) for elem in Count_sec])
        return str(B) + "A" + str(B_C-B) + "B"

# use Counter to count guess and secret and sum their overlap. 
# Then use zip to count A.
class Solution(object):
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """
        s, g = Counter(secret), Counter(guess)
        a = sum(i == j for i, j in zip(secret, guess))
        return '%sA%sB' % (a, sum((s & g).values()) - a)
# Day 14 Stack
# 844. Backspace String Compare

# 394. Decode String
class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []; curNum = 0; curString = ''
        for c in s:
            if c == '[':
                stack.append(curString)
                stack.append(curNum)
                curString = ''
                curNum = 0
            elif c == ']':
                num = stack.pop()
                prevString = stack.pop()
                curString = prevString + num*curString
            elif c.isdigit():
                curNum = curNum*10 + int(c)
            else:
                curString += c
        return curString

# Day 15 Heap
# 1046. Last Stone Weight
'''
Priority Queue
Explanation
Put all elements into a priority queue.
Pop out the two biggest, push back the difference,
until there are no more two elements left.


Complexity
Time O(NlogN)
Space O(N)
'''
# using heap, O(NlogN) time
class Solution(object):
    def lastStoneWeight(self, stones):
        """
        :type stones: List[int]
        :rtype: int
        """
        h = [-x for x in stones]
        heapq.heapify(h)
        while len(h) > 1 and h[0] != 0:
            heapq.heappush(h, heapq.heappop(h) - heapq.heappop(h))
        return -h[0]
# using binary insort, O(N^2) time
class Solution(object):
    def lastStoneWeight(self, stones):
        """
        :type stones: List[int]
        :rtype: int
        """
        stones.sort()
        while len(stones) > 1:
            bisect.insort(stones, stones.pop() - stones.pop())
        return stones[0]
# 692. Top K Frequent Words
'''
Here's a 95% speed solution.
(1) Use collections.Counter to collate
(2) create list of touples containing (word,count)
(3) Sort by the count and secondary by the word. Note that by negating the count we sort from highest count to lowest instead of the other way around. (Note also that you can't just do a reverse sort or the words themselves would be the wrong way around.)
(4) Strip off and return a list of the first K words.'''
class Solution(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        counts = collections.Counter(words)
        items = list(counts.items())
        items.sort(key=lambda item:(-item[1],item[0]))
        return [item[0] for item in items[0:k]]