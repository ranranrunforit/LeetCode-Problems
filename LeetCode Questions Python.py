# 2420. Find All Good Indices

# 2419. Longest Subarray With Maximum Bitwise AND

# 2418. Sort the People

# 2416. Sum of Prefix Scores of Strings
'''
Build Trie and accumulate the frequencies of each pefix at the same time; 
then search each word and compute the corresponding score.
Analysis:

Time & space: O(n * w), where n = words.length,
 w = average size of word in `words``.'''
class Trie:
    def __init__(self):
        self.cnt = 0
        self.kids = {}
    def add(self, word):
        trie = self
        for c in word:
            if c not in trie.kids:
                trie.kids[c] = Trie()
            trie = trie.kids[c]
            trie.cnt += 1
    def search(self, word):
        score = 0
        trie = self
        for c in word:
            if c not in trie.kids:
                return score
            trie = trie.kids[c]        
            score += trie.cnt
        return score
    
class Solution(object):
    def sumPrefixScores(self, words):
        """
        :type words: List[str]
        :rtype: List[int]
        """
        root = Trie()
        for w in words:
            root.add(w)
        return [root.search(w) for w in words]

# 2410. Maximum Matching of Players With Trainers

# 2409. Count Days Spent Together

# 2407. Longest Increasing Subsequence II

# 2405. Optimal Partition of String

# 2404. Most Frequent Even Element

# 2399. Check Distances Between Same Letters

# 2396. Strictly Palindromic Number
class Solution(object):
    def isStrictlyPalindromic(self, n):
        """
        :type n: int
        :rtype: bool
        """
        return False
# 2395. Find Subarrays With Equal Sum

# 133. Clone Graph
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""
# BFS
class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if not node:
            return node
        m, visited, stack = dict(), set(), deque([node])
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            if n not in m:
                m[n] = Node(n.val)
            for neigh in n.neighbors:
                if neigh not in m:
                    m[neigh] = Node(neigh.val)
                m[n].neighbors.append(m[neigh])
                stack.append(neigh)
        return m[node]
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""
# dfs iteratively
class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if not node:
            return node
        m, visited, queue = {}, set(), collections.deque([node])
        while queue:
            n = queue.popleft()
            if n in visited:
                continue
            visited.add(n)
            if n not in m:
                m[n] = Node(n.val)
            for neigh in n.neighbors:
                if neigh not in m:
                    m[neigh] = Node(neigh.val)
                m[n].neighbors.append(m[neigh])
                queue.append(neigh)
        return m[node]

"""
# Definition for a Node.
class Node(object):
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""
# dfs recursively 
class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if not node:
            return node
        m, visited = dict(), set()
        self.dfs(node, m, visited)
        return m[node]
        
    def dfs(self, n, m, visited):
        if n in visited:
            return 
        visited.add(n)
        if n not in m:
            m[n] = Node(n.val)
        for neigh in n.neighbors:
            if neigh not in m:
                m[neigh] = Node(neigh.val)
            m[n].neighbors.append(m[neigh])
            self.dfs(neigh, m, visited)
        
# 2317. Maximum XOR After Operations
'''Explanation
The maximum possible result is res = A[0] || A[1] || A[2] ... and it's realisiable.

Prove
Now we approve it's realisiable.
Assume result is best = XOR(A[i]) and best < res above.
There is at least one bit difference between best and res, assume it's x = 1 << k.

We can find at least a A[i] that A[i] & x = x.

we apply x on A[i], A[i] is updated to A[i] & (A[i] ^ x) = A[i] ^ x.
We had best = XOR(A[i]) as said above,
now we have best2 = XOR(A[i]) ^ x,
so we get a better best2 > best, where we prove by contradiction.


Complexity
Time O(n)
Space O(1)'''
class Solution(object):
    def maximumXOR(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return reduce(ior, nums)

# 2374. Node With Highest Edge Score
class Solution(object):
    def edgeScore(self, edges):
        """
        :type edges: List[int]
        :rtype: int
        """
        n, ans = len(edges), 0
        scores = [0] * n
        for src, tgt in enumerate(edges):
            scores[tgt] += src
        for i, score in enumerate(scores):
            if score > scores[ans]:
                ans = i
        return ans
# 2265. Count Nodes Equal to Average of Subtree

# 2260. Minimum Consecutive Cards to Pick Up

# 2264. Largest 3-Same-Digit Number in String

# 2266. Count Number of Texts
# Python3 only
'''
Solution: DP
Similar to 花花酱 LeetCode 91. Decode Ways, let dp[i] denote # of possible messages of substr s[i:]

dp[i] = dp[i + 1]
+ dp[i + 2] (if s[i:i+1] are the same)
+ dp[i + 3] (if s[i:i+2] are the same)
+ dp[i + 4] (if s[i:i+3] are the same and s[i] in ’79’)

dp[n] = 1

Time complexity: O(n)
Space complexity: O(n) -> O(4)
'''
class Solution:
    def countTexts(self, pressedKeys: str) -> int:
        kMod = 10**9 + 7
        n = len(pressedKeys)

        @cache
        def dp(i: int) -> int:      
            if i >= n: return 1
            ans = dp(i + 1)
            ans += dp(i + 2) if i + 2 <= n and pressedKeys[i] == pressedKeys[i + 1] else 0
            ans += dp(i + 3) if i + 3 <= n and pressedKeys[i] == pressedKeys[i + 1] == pressedKeys[i + 2] else 0
            ans += dp(i + 4) if i + 4 <= n and pressedKeys[i] == pressedKeys[i + 1] == pressedKeys[i + 2] == pressedKeys[i + 3] and pressedKeys[i] in '79' else 0      
            return ans % kMod

        return dp(0)

# 2267. Check if There Is a Valid Parentheses String Path
# Python3 only
'''
Solution: DP
Let dp(i, j, b) denote whether there is a path from (i,j) to (m-1, n-1) given b open parentheses.
if we are at (m – 1, n – 1) and b == 0 then we found a valid path.
dp(i, j, b) = dp(i + 1, j, b’) or dp(i, j + 1, b’) where b’ = b + 1 if grid[i][j] == ‘(‘ else -1

Time complexity: O(m*n*(m + n))
Space complexity: O(m*n*(m + n))
'''
class Solution:
    def hasValidPath(self, grid: List[List[str]]) -> bool:
        m, n = len(grid), len(grid[0])    
    
        @cache
        def dp(i: int, j: int, b: int) -> bool:
            if b < 0 or i == m or j == n: return False
            b += 1 if grid[i][j] == '(' else -1      
            if i == m - 1 and j == n - 1 and b == 0: return True      
            return dp(i + 1, j, b) or dp(i, j + 1, b)

        return dp(0, 0, 0)

# 2304. Minimum Path Cost in a Grid

# 2303. Calculate Amount Paid in Taxes

# 2315. Count Asterisks
'''
Explanation
Parse the input, if currently met odd bars, we count *.

Complexity
Time O(n)
Space O(1)
'''
class Solution(object):
    def countAsterisks(self, s):
        """
        :type s: str
        :rtype: int
        """
        return sum([a.count('*') for a in s.split('|')][0::2])

# 2269. Find the K-Beauty of a Number

# 2270. Number of Ways to Split Array

# 2251. Number of Flowers in Full Bloom
'''Solution 1: Binary Seach
Intuition
Blooming flowers = started flowers - ended flowers

Explanation
Collect start bloom time point array, then sort it.
Collect end bloom time point array, then sort it.

For each time point t in persons:

Binary search the upper bound of t in start, then we find the started flowers.
Binary search the lower bound of t in end, then we find the started flowers.
Blooming flowers = started flowers - ended flowers

Complexity
Time O(nlogn + mlogn)
Space O(n)'''
class Solution(object):
    def fullBloomFlowers(self, flowers, persons):
        """
        :type flowers: List[List[int]]
        :type persons: List[int]
        :rtype: List[int]
        """
        start, end = sorted(a for a,b in flowers), sorted(b for a,b in flowers)
        return [bisect_right(start, t) - bisect_left(end, t) for t in persons]

# 2255. Count Prefixes of a Given String
'''
Explanation
for each word w in words list,
check if word w startsWith the string s


Complexity
Time O(NS)
Space O(1)'''
class Solution(object):
    def countPrefixes(self, words, s):
        """
        :type words: List[str]
        :type s: str
        :rtype: int
        """
        return sum(map(s.startswith, words))

# 2242. Maximum Score of a Node Sequence
'''
Intuition
We don't need to check all possible sequences,
but only some big nodes.


Explanation
For each edge (i, j) in edges,
we find a neighbour ii of node i,
we find a neighbour jj of node i,
If ii, i, j,jj has no duplicate, then that's a valid sequence.

Ad the intuition mentioned,
we don't have to enumearte all neignbours,
but only some nodes with big value.

But how many is enough?
I'll say 3.
For example, we have ii, i, j now,
we can enumerate 3 of node j biggest neighbour,
there must be at least one node different node ii and node i.

So we need to iterate all edges (i, j),
for each node we keep at most 3 biggest neighbour, which this can be done in O(3) or O(log3).


Complexity
Time O(n + m)
Space O(n + m)
'''
class Solution(object):
    def maximumScore(self, scores, edges):
        """
        :type scores: List[int]
        :type edges: List[List[int]]
        :rtype: int
        """
        n = len(scores)
        G = [[] for i in range(n)]
        for i,j in edges:
            G[i].append([scores[j], j])
            G[j].append([scores[i], i])
        for i in range(n):
            G[i] = nlargest(3, G[i])
            
        res = -1
        for i,j  in edges:
            for vii, ii in G[i]:
                for vjj, jj in G[j]:
                    if ii != jj and ii != j and i != jj:
                    #if ii != jj and ii != j and j != ii:
                        res = max(res, vii + vjj + scores[i] + scores[j])
        return res

# 2243. Calculate Digit Sum of a String

# 2244. Minimum Rounds to Complete All Tasks
'''
Intuition
If the frequence freq of a level is 1,
then it is not possible to complete all the tasks.

Otherwise, we need to decompose freq = 3 tasks + 3 tasks + .... + 2 tasks,
with the minimum number of 3 and 2.

We need a lot a 3-tasks, and plus one or two 2-tasks.


Explanation
Tasks with same difficulty level can be done together,
in group of 2-tasks or 3-tasks.

So we count the frequnce freq for each level.

If freq = 1, not possible, return -1
If freq = 2, needs one 2-tasks
If freq = 3, needs one 3-tasks
If freq = 3k, freq = 3 * k, needs k batchs.
If freq = 3k + 1, freq = 3 * (k - 1) + 2 + 2, needs k + 1 batchs.
If freq = 3k + 2, freq = 3 * k + 2, needs k + 1 batchs.

To summarize, needs (freq + 2) / 3 batch,
return the sum of (freq + 2) / 3 if possible.


Complexity
Time O(n)
Space O(n)'''
class Solution(object):
    def minimumRounds(self, tasks):
        """
        :type tasks: List[int]
        :rtype: int
        """
        freq = Counter(tasks).values()
        return -1 if 1 in freq else sum((a + 2) # 3 for a in freq)

# 2248. Intersection of Multiple Arrays

# 2249. Count Lattice Points Inside a Circle
'''
Updated this post,
since I noticed almost every solutions will use set,
with extra space complexity.

Solution 1: Set
Explanation
For each circle (x, y),
enumerate i from x - r to x + r
enumerate j from y - r to y + r
Check if (i, j) in this circle.

If so, add the point to a set res to de-duplicate.

Complexity
Time O(NRR)
Space O(XY)
where R = 100 is maximum radius of circles.

Actual run time depends on circles area in test cases.
'''
class Solution(object):
    def countLatticePoints(self, circles):
        """
        :type circles: List[List[int]]
        :rtype: int
        """
        res = set()
        for x,y,r in circles:
            for i in range(x - r,x + r + 1):
                for j in range(y - r, y + r + 1):
                    if (x - i) ** 2 + (y - j) ** 2 <= r * r:
                        res.add((i,j))
        return len(res)

# 2241. Design an ATM Machine

# 2240. Number of Ways to Buy Pens and Pencils
'''
Solution:
Enumerate all possible ways to buy k pens, e.g. 0 pen, 1 pen, …, total / cost1.
The way to buy pencils are (total – k * cost1) / cost2 + 1.
ans = sum((total – k * cost1) / cost2 + 1)) for k = 0 to total / cost1.

Time complexity: O(total / cost1)
Space complexity: O(1)
'''
class Solution(object):
    def waysToBuyPensPencils(self, total, cost1, cost2):
        """
        :type total: int
        :type cost1: int
        :type cost2: int
        :rtype: int
        """
        ans = 0
        for k in range(0, (total # cost1) + 1 ):
            ans += ((total - k * cost1) # cost2) + 1
        return ans
    
# 2239. Find Closest Number to Zero

# 2236. Root Equals Sum of Children 
'''
Solution:
Just want to check whether you know binary tree or not.

Time complexity: O(1)
Space complexity: O(1)'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def checkTree(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        return root.val == root.left.val + root.right.val

# 2235. Add Two Integers
'''
Solution: Just sum them up
Time complexity: O(1)
Space complexity: O(1)
'''
class Solution(object):
    def sum(self, num1, num2):
        """
        :type num1: int
        :type num2: int
        :rtype: int
        """
        return num1 + num2

# 2233. Maximum Product After K Increments
class Solution(object):
    def maximumProduct(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        heapify(nums)
        for i in range(k):
            heappush(nums, heappop(nums) + 1)
        return reduce(lambda p, a: p * a % (10**9 + 7), nums, 1)

# 2232. Minimize Result by Adding Parentheses to Expression

# 2231. Largest Number After Digit Swaps by Parity

# 2227. Encrypt and Decrypt Strings
'''
Python
3-lines Python.
decrypt method is not missing, defined in init
I remoded unnecessay code from default template.'''
class Encrypter(object):

    def __init__(self, keys, values, dictionary):
        """
        :type keys: List[str]
        :type values: List[str]
        :type dictionary: List[str]
        """
        self.enc = {k: v for k,v in zip(keys, values)}
        self.decrypt = collections.Counter(self.encrypt(w) for w in dictionary).__getitem__

    def encrypt(self, word1):
        """
        :type word1: str
        :rtype: str
        """
        return ''.join(self.enc.get(c, '#') for c in word1)

    def decrypt(self, word2):
        """
        :type word2: str
        :rtype: int
        """
        


# Your Encrypter object will be instantiated and called as such:
# obj = Encrypter(keys, values, dictionary)
# param_1 = obj.encrypt(word1)
# param_2 = obj.decrypt(word2)

# 2226. Maximum Candies Allocated to K Children
'''
Intuition
Binary search


Explanation
Assume we want give each child m candies, for each pile of candies[i],
we can divide out at most candies[i] / m sub piles with each pile m candies.

We can sum up all the sub piles we can divide out, then compare with the k children.

If k > sum,
we don't allocate to every child,
since the pile of m candidies it too big,
so we assign right = m - 1.

If k <= sum,
we are able to allocate to every child,
since the pile of m candidies is small enough
so we assign left = m.

We repeatly do this until left == right, and that's the maximum number of candies each child can get.


Tips
Tip1. left < right Vs left <= right

Check all my solution, I keep using left < right.
The easy but important approach:
follow and upvote my codes,
try to do the same.
you'll find all binary search is similar,
never bother thinking it anymore.

Tip2. mid = (left + right + 1) / 2 Vs mid = (left + right) / 2

mid = (left + right) / 2 to find first element valid
mid = (left + right + 1) / 2to find last element valid


Complexity
Time O(nlog10000000)
Space O(1)'''
class Solution(object):
    def maximumCandies(self, candies, k):
        """
        :type candies: List[int]
        :type k: int
        :rtype: int
        """
        left, right = 0, sum(candies) / k
        while left < right:
            mid = (left + right + 1) / 2
            if k > sum(a / mid for a in candies):
                right = mid - 1
            else:
                left = mid
        return left

# 2239. Find Closest Number to Zero
'''
Explanation
Fine the max pair of (-abs(a), a),
where a is the elements in input array.

It compare firstly -abs(a), where it finds the minimum absolute value.
If multiple result, then it compares secondely a, where it finds the maximum value.

Complexity
Time O(n)
Space O(1)'''
class Solution(object):
    def findClosestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return max([-abs(a), a] for a in nums)[1] 

class Solution(object):
    def findClosestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        best = nums[0]
        for n in nums:
            if abs(n-0.1) < abs(best-0.1):
                best = n
        return best
# 2225. Find Players With Zero or One Losses
# 2224. Minimum Number of Operations to Convert Time
# 2223. Sum of Scores of Built Strings
# 2222. Number of Ways to Select Buildings
# 2221. Find Triangular Sum of an Array
# 2220. Minimum Bit Flips to Convert Number
# 2218. Maximum Value of K Coins From Piles
'''
Solution: DP
let dp(i, k) be the maximum value of picking k elements using piles[i:n].

dp(i, k) = max(dp(i + 1, k), sum(piles[i][0~j]) + dp(i + 1, k – j – 1)), 0 <= j < len(piles[i])

Time complexity: O(n * m), m = sum(piles[i]) <= 2000
Space complexity: O(n * k)


'''
class Solution:
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        n = len(piles)
    
        @cache
        def dp(i: int, k: int) -> int:
            """Max value of picking k elements using piles[i:n]."""
            if i == n: return 0
            ans, cur = dp(i + 1, k), 0      
            for j in range(min(len(piles[i]), k)):        
                ans = max(ans, (cur := cur + piles[i][j]) + dp(i + 1, k - j - 1))
            return ans

        return dp(0, k)
# 2216. Minimum Deletions to Make Array Beautiful
# 2215. Find the Difference of Two Arrays
class Solution(object):
    def findDifference(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[List[int]]
        """
        s1, s2 = set(nums1), set(nums2)
        return [list(s1 - s2), list(s2 - s1)]
        
# 2206. Divide Array Into Equal Pairs
class Solution(object):
    def divideArray(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        return all(v % 2 == 0 for v in Counter(nums).values())

# 2197. Replace Non-Coprime Numbers in Array
'''
Explanation
For each number a in input array A,
check if it is coprime with the last number b in res.
If it's not coprime, then we can merge them by calculate a * b / gcd(a, b).
and check we can continue to do this process.

Until it's coprime with the last element in res,
we append a at the end of res.

We do this for all elements a in A, and return the final result.


Complexity
Time O(nlogn)
Space O(n)

'''
class Solution:
    def replaceNonCoprimes(self, nums: List[int]) -> List[int]:
        res = []
        for a in nums:
            while True:
                x = math.gcd(res[-1] if res else 1, a)
                if x == 1: break # co-prime
                a *= res.pop() # x
            res.append(a)
        return res

# 2196. Create Binary Tree From Descriptions
'''
Explanation
Iterate descriptions,
for each [p, c, l] of [parent, child, isLeft]

Create Treenode with value p and c,
and store them in a hash map with the value as key,
so that we can access the TreeNode easily.

Based on the value isLeft,
we assign Treenode(parent).left = Treenode(child)
or Treenode(parent).right = Treenode(child).

Finall we find the root of the tree, and return its TreeNode.
'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def createBinaryTree(self, descriptions):
        """
        :type descriptions: List[List[int]]
        :rtype: Optional[TreeNode]
        """
        children = set()
        m = {}
        for p,c,l in descriptions:
            np = m.setdefault(p, TreeNode(p))
            nc = m.setdefault(c, TreeNode(c))
            if l:
                np.left = nc
            else:
                np.right = nc
            children.add(c)
        root = (set(m) - set(children)).pop()
        return m[root]
# 2195. Append K Integers With Minimal Sum

# 2194. Cells in a Range on an Excel Sheet

# 2178. Maximum Split of Positive Even Integers

# 2177. Find Three Consecutive Integers That Sum to a Given Number

# 2176. Count Equal and Divisible Pairs in an Array
# 2180. Count Integers With Even Digit Sum
# 2181. Merge Nodes in Between Zeros
# 2182. Construct String With Repeat Limit
# 2183. Count Array Pairs Divisible by K
# 2146. K Highest Ranked Items Within a Price Range
''' negating the grid values to using HashSet'''
class Solution(object):
    def highestRankedKItems(self, grid, pricing, start, k):
        """
        :type grid: List[List[int]]
        :type pricing: List[int]
        :type start: List[int]
        :type k: int
        :rtype: List[List[int]]
        """
        R, C = map(len, (grid, grid[0]))
        ans, (x, y), (low, high) = [], start, pricing
        heap = [(0, grid[x][y], x, y)]
        seen = {(x, y)}
        while heap and len(ans) < k:
            distance, price, r, c = heapq.heappop(heap)
            if low <= price <= high:
                ans.append([r, c])
            for i, j in (r, c + 1), (r, c - 1), (r + 1, c), (r - 1, c):
                if R > i >= 0 <= j < C and grid[i][j] > 0 and (i, j) not in seen: 
                    seen.add((i, j))
                    heapq.heappush(heap, (distance + 1, grid[i][j], i, j))
        return ans
'''
Use a PriorityQueue/heap to store (distance, price, row, col) and 
a HashSet to prune duplicates.
The PriorityQueue/heap is sorted according to the rank specified in the problem. 
Therefore, the ans always holds highest ranked items within the given price range, 
and once it reaches the size of k, the loop does NOT need to iterate any more.'''       
class Solution(object):
    def highestRankedKItems(self, grid, pricing, start, k):
        """
        :type grid: List[List[int]]
        :type pricing: List[int]
        :type start: List[int]
        :type k: int
        :rtype: List[List[int]]
        """
        R, C = map(len, (grid, grid[0]))
        ans, (x, y), (low, high) = [], start, pricing
        heap = [(0, grid[x][y], x, y)]
        grid[x][y] *= -1
        while heap and len(ans) < k:
            distance, price, r, c = heapq.heappop(heap)
            if low <= price <= high:
                ans.append([r, c])
            for i, j in (r, c + 1), (r, c - 1), (r + 1, c), (r - 1, c):
                if R > i >= 0 <= j < C and grid[i][j] > 0: 
                    heapq.heappush(heap, (distance + 1, grid[i][j], i, j))
                    grid[i][j] *= -1
        return ans
# 2144. Minimum Cost of Buying Candies With Discount
'''Explanation
For the max value, we have to pay for it.
For the second max value, we still have to pay for it.
For the third max value, we can get it free one as bonus.
And continuely do this for the rest.

The the core of problem, is need to sort the input.
All A[i] with i % 3 == n % 3, we can get it for free.


Complexity
Time O(sort)
Space O(sort)'''
class Solution(object):
    def minimumCost(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        return sum(cost) - sum(sorted(cost)[-3::-3])

class Solution(object):
    def minimumCost(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        return sum(a for i,a in enumerate(sorted(cost)) if (len(cost) - i) % 3)

# 2151. Maximum Good People Based on Statements
class Solution(object):
    def maximumGood(self, A):
        """
        :type statements: List[List[int]]
        :rtype: int
        """
        n, ans = len(A), 0
    
        def check(perm):
            for i in range(n):
                if perm[i] == '0': continue
                for j in range(n):
                    if A[i][j] == 2: continue
                    if (A[i][j] == 1 and perm[j] == '0') or (A[i][j] == 0 and perm[j] == '1'): 
                        return False
            return True

        for num in range(1 << n, 1 << (n + 1)):
            permutation = bin(num)[3:]
            if check(permutation): 
                ans = max(ans, permutation.count('1'))
        return ans

# 2156. Find Substring With Given Hash Value
'''Sliding Window + Rolling Hash
Intuition
Good time to learn rolling hash.
what's hash?
The definition hash(s, p, m) in the description is the hash of string s based on p.

what's rolling hash?
The hash of substring is a sliding window.
So the basis of rolling hash is sliding window.

Explanation
Calculate the rolling hash backward.
In this process, we slide a window of size k from the end to the begin.

Firstly calculate the substring hash of the last k characters,
then we add one previous backward and drop the last characters.

Why traverse from end instead of front?
Because cur is reminder by mod m,
cur = cur * p works easier.
cur = cur / p doesn'r work easily.


Complexity
Time O(n)
Space O(1)
'''
class Solution(object):
    def subStrHash(self, s, p, m, k, hashValue):
        """
        :type s: str
        :type power: int
        :type modulo: int
        :type k: int
        :type hashValue: int
        :rtype: str
        """
        def val(c):
            return ord(c) - ord('a') + 1
            
        res = n = len(s)
        pk = pow(p,k,m)
        cur = 0

        for i in xrange(n - 1, -1, -1):
            cur = (cur * p + val(s[i])) % m
            if i + k < n:
                cur = (cur - val(s[i + k]) * pk) % m
            if cur == hashValue:
                res = i
        return s[res: res + k]

# 2157. Groups of Strings
from typing import List

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.sz = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py
            self.sz[py] += self.sz[px]
            self.sz[px] = 0
            
class Solution:
    def groupStrings(self, words: List[str]) -> List[int]:
        n, uf = len(words), UnionFind(len(words))
        dt = dict()

        for i, w in enumerate(words):
            x = sum(1 << (ord(c) - ord('a')) for c in w)
            if x in dt:
                uf.union(i, dt[x])

            # if 2 words are connected by replacement, they have same child after deleting 1 letter
            for j in range(26):
                if x & (1 << j):
                    y = x ^ (1 << j)
                    if y in dt:
                        uf.union(i, dt[y])
                    dt[y] = i
            dt[x] = i

        for i in range(n):
            uf.find(i)

        return [len(set(uf.parent)), max(uf.sz)]

# 2140. Solving Questions With Brainpower
'''Solution: DP
A more general version of 花花酱 LeetCode 198. House Robber

dp[i] := max points by solving questions[i:n].
dp[i] = max(dp[i + b + 1] + points[i] /* solve */ , dp[i+1] /* skip */)

ans = dp[0]

Time complexity: O(n)
Space complexity: O(n)'''
class Solution:
    def mostPoints(self, questions: List[List[int]]) -> int:
        n = len(questions)
        @cache
        def dp(i: int) -> int:
            if i >= n: return 0
            p, b = questions[i]      
            return max(p + dp(i + b + 1), dp(i + 1))

        return dp(0)

# 2139. Minimum Moves to Reach Target Score
'''
Intuition
We should use double action as late as possible, as many as possible.
Do this process reversely: Reduce target to 1.
We try to use the HALF action as soon as possbile..


Explanation
If we still "half" action, we do it.
If it's odd, we decrese it by 1 and make it half, which takes two actions.
If it's even, we make it half, which takes one action.

If no more "half" action, we decrement continuously to 1, which takes target - 1 actions.


Complexity
Time O(logn)
Space O(1)'''
class Solution(object):
    def minMoves(self, target, maxDoubles):
        """
        :type target: int
        :type maxDoubles: int
        :rtype: int
        """
        res = 0
        while target > 1 and maxDoubles > 0:
            res += 1 + target % 2
            maxDoubles -= 1
            target >>= 1
        return target - 1 + res

#  2135. Count Words Obtained After Adding a Letter
class Solution(object):
    def wordCount(self, startWords, targetWords):
        """
        :type startWords: List[str]
        :type targetWords: List[str]
        :rtype: int
        """
        seen = set()
        for word in startWords: 
            m = 0
            for ch in word: m ^= 1 << ord(ch)-97
            seen.add(m)
            
        ans = 0 
        for word in targetWords: 
            m = 0 
            for ch in word: m ^= 1 << ord(ch)-97
            for ch in word: 
                if m ^ (1 << ord(ch)-97) in seen: 
                    ans += 1
                    break 
        return ans 

# 2129. Capitalize the Title
'''Solution: Straight forward
Without splitting the sentence into words, we need to take care the word of length one and two.

Tips: use std::tolower, std::toupper to transform letters.

Time complexity: O(n)
Space complexity: O(1)'''
class Solution(object):
    def capitalizeTitle(self, title):
        """
        :type title: str
        :rtype: str
        """
        return " ".join(map(lambda w: w.lower() if len(w) <= 2 else w[0].upper() + w[1:].lower(), title.split()))
        
#  1995. Count Special Quadruplets
'''a + b + c = d = > a + b = d - c
Break array into two parts[0, i - 1] and [i, n -1]
for each i,
step 1: calculate all possible d - c and put them in a hashMap called diffCount .
d - c = nums[j] - nums[i]. for all j [i + 1, n - 1]
step 2: calculate all possible a + b in the 1st part. Then check if any a + b in the hashMap diffCount
a + b = nums[j] + nums[i - 1], for all j [0, i - 2]

Time complexity: O(n^2)
Space complexity: O(n^2)'''
class Solution(object):
    def countQuadruplets(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        diffCount = {}
        result = 0
        n = len(nums)
        for i in range(n - 2, 0, -1):
            for j in range(i + 1, n):
                num = nums[j] - nums[i]
                diffCount[num] = diffCount.get(num, 0) + 1
            
            for j in range(i - 2, -1, -1):
                num = nums[j] + nums[i - 1]
                result += diffCount.get(num, 0)
        return result
        
# 2125. Number of Laser Beams in a Bank
class Solution(object):
    def numberOfBeams(self, bank):
        """
        :type bank: List[str]
        :rtype: int
        """
        ans = prev = 0
        for s in bank:
            c = s.count('1')
            if c:
                ans += prev * c
                prev = c
        return ans

# 1975. Maximum Matrix Sum
# 1974. Minimum Time to Type Word Using Special Typewriter
# 1967. Number of Strings That Appear as Substrings in Word
# 1962. Remove Stones to Minimize the Total
'''
Heap Solution, O(klogn)
Explanation
Use a max heap.
Each time pop the max value a,
remove a / 2 from the number of stones res
and push back the ceil half a - a / 2 to the heap.
Repeat this operation k times.


Complexity
Time O(n + klogn)
Space O(n)
'''
class Solution(object):
    def minStoneSum(self, A, k):
        """
        :type piles: List[int]
        :type k: int
        :rtype: int
        """
        A = [-a for a in A]
        heapq.heapify(A)
        for i in xrange(k):
            heapq.heapreplace(A, A[0] / 2)
        return -sum(A)

# 1961. Check If String Is a Prefix of Array

# 1957. Delete Characters to Make Fancy String

# 1953. Maximum Number of Weeks for Which You Can Work

# 1952. Three Divisors

# 1946. Largest Number After Mutating Substring

# 1945. Sum of Digits of String After Convert

# 1942. The Number of the Smallest Unoccupied Chair

# 1941. Check if All Characters Have Equal Number of Occurrences
'''
Solution: Hashtable
Time complexity: O(n)
Space complexity: O(1)'''
class Solution(object):
    def areOccurrencesEqual(self, s):
        """
        :type s: str
        :rtype: bool
        """
        return len(set(Counter(s).values())) == 1

# 1936. Add Minimum Number of Rungs
'''
Solution: Math
Check two consecutive rungs, if their diff is > dist, we need insert (diff – 1) / dist rungs in between.
ex1 5 -> 11, diff = 6, dist = 2, (diff – 1) / dist = (6 – 1) / 2 = 2. => 5, 7, 9, 11.
ex2 0 -> 3, diff = 3, dist = 1, (diff – 1) / dist = (3 – 1) / 1 = 2 => 0, 1, 2, 3

Time complexity: O(n)
Space complexity: O(1)

'''
class Solution(object):
    def addRungs(self, rungs, dist):
        """
        :type rungs: List[int]
        :type dist: int
        :rtype: int
        """
        return sum((r - 1 - (rungs[i - 1] if i else 0)) # dist for i, r in enumerate(rungs))

# 1935. Maximum Number of Words You Can Type
# Method 1: HashSet
class Solution(object):
    def canBeTypedWords(self, text, brokenLetters):
        """
        :type text: str
        :type brokenLetters: str
        :rtype: int
        """
        no, cnt = set(brokenLetters), 0
        for word in text.split():
            if all(c not in no for c in word):
                cnt += 1
        return cnt
# Method 2: Bit Manipulation
class Solution(object):
    def canBeTypedWords(self, text, brokenLetters):
        """
        :type text: str
        :type brokenLetters: str
        :rtype: int
        """
        mask = functools.reduce(lambda x, y: x | 1 << ord(y) - ord('a'), brokenLetters, 0)
        return sum(1 for word in text.split() if all(((1 << ord(c) - ord('a')) & mask) == 0 for c in word))

# 1930. Unique Length-3 Palindromic Subsequences
'''Straight Forward Solution
Explanation
For each palindromes in format of "aba",
we enumerate the character on two side.

We find its first occurrence and its last occurrence,
all the characters in the middle are the candidate for the midd char.


Complexity
Time O(26n)
Space O(26n)'''
class Solution(object):
    def countPalindromicSubsequence(self, s):
        """
        :type s: str
        :rtype: int
        """
        res = 0
        for c in string.ascii_lowercase:
            i, j = s.find(c), s.rfind(c)
            if i > -1:
                res += len(set(s[i + 1: j]))
        return res

# 1929. Concatenation of Array

# 1925. Count Square Sum Triples

# 1922. Count Good Numbers
'''Analysis:

Time: O(logn), space: O(1).'''
class Solution(object):
    def countGoodNumbers(self, n):
        """
        :type n: int
        :rtype: int
        """
        MOD = 10 ** 9 + 7
        good, x, i = 5 ** (n % 2), 4 * 5, n # 2
        while i > 0:
            if i % 2 == 1:
                good = good * x % MOD
            x = x * x % MOD
            i #= 2
        return good

# 1921. Eliminate Maximum Number of Monsters

# 1920. Build Array from Permutation

# 2122. Recover the Original Array
class Solution(object):
    def recoverArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        def check(nums, k):
            cnt, ans = Counter(nums), []
            for num in nums:
                if cnt[num] == 0: continue
                if cnt[num + k] == 0: return False, []
                cnt[num] -= 1
                cnt[num + k] -= 1
                ans += [num + k#2]
            return True, ans
            
        nums = sorted(nums)
        n = len(nums)
        for i in range(1, n):
            k = nums[i] - nums[0]
            if k != 0 and k % 2 == 0:
                a, b = check(nums, k)
                if a: return b

# 2121. Intervals Between Identical Elements
''' Python 3 prefix sum
Explanation on the formula
Given an array, say nums = [3,2,1,5,4]. Calculate its prefix sum [0,3,5,6,11,15] (note the leading zero). 
Now let's loop through nums by index. At i, there are

i numbers whose indices are below i with sum prefix[i];
len(nums)-i-1 numbers whose indices are above i with sum prefix[-1] - prefix[i+1].
So the desired value in this case is i*x - prefix[i] + prefix[-1] - prefix[i+1] - (len(nums)-i-1)*x. 
Rearranging the terms one would arrive at the formula used in this implementation.
'''
class Solution:
    def getDistances(self, arr: List[int]) -> List[int]:
        loc = defaultdict(list)
        for i, x in enumerate(arr): loc[x].append(i)
        
        for k, idx in loc.items(): 
            prefix = list(accumulate(idx, initial=0))
            vals = []
            for i, x in enumerate(idx): 
                vals.append(prefix[-1] - prefix[i] - prefix[i+1] - (len(idx)-2*i-1)*x)
            loc[k] = deque(vals)
        
        return [loc[x].popleft() for x in arr]

# 2120. Execution of All Suffix Instructions Staying in a Grid

# 2119. A Number After a Double Reversal
'''Solution: Math
The number must not end with 0 expect 0 itself.

e.g. 1230 => 321 => 123
e.g. 0 => 0 => 0

Time complexity: O(1)
Space complexity: O(1)'''
class Solution(object):
    def isSameAfterReversals(self, num):
        """
        :type num: int
        :rtype: bool
        """
        return num == 0 or num % 10  # num % 10 means num % 10 != 0

# 2117. Abbreviating the Product of a Range
'''
Solution: Prefix + Suffix
Since we only need the first 5 digits and last 5 digits, we can compute prefix and suffix separately with 15+ effective digits. Note, if using long/int64 with (18 – 6) = 12 effective digits, it may fail on certain test cases. Thus, here we use Python with 18 effective digits.

Time complexity: O(mlog(right)) where m = right – left + 1
Space complexity: O(1)

'''
class Solution(object):
    def abbreviateProduct(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: str
        """
        kMax = 10 ** 18
        prefix = 1
        suffix = 1
        c = 0
        for i in range(left, right + 1):
            prefix *= i
            while prefix >= kMax:
                prefix #= 10
            suffix *= i
            while suffix % 10 == 0: 
                suffix #= 10
                c += 1
            suffix %= kMax

        p, s = str(prefix), str(suffix)
        return (s if len(s) <= 10 else p[:5] + "..." + s[-5:]) + "e" + str(c)

# 2115. Find All Possible Recipes from Given Supplies
class Solution(object):
    def findAllRecipes(self, recipes, ingredients, supplies):
        """
        :type recipes: List[str]
        :type ingredients: List[List[str]]
        :type supplies: List[str]
        :rtype: List[str]
        """
        indeg = defaultdict(int)
        graph = defaultdict(list)
        for r, ing in zip(recipes, ingredients): 
            indeg[r] = len(ing)
            for i in ing: graph[i].append(r)
        
        ans = []
        queue = deque(supplies)
        recipes = set(recipes)
        while queue: 
            x = queue.popleft()
            if x in recipes: ans.append(x)
            for xx in graph[x]: 
                indeg[xx] -= 1
                if indeg[xx] == 0: queue.append(xx)
        return ans 

# 2114. Maximum Number of Words Found in Sentences
class Solution(object):
    def mostWordsFound(self, sentences):
        """
        :type sentences: List[str]
        :rtype: int
        """
        return max(len(s.split()) for s in sentences)

# 2111. Minimum Operations to Make the Array K-Increasing
'''
Solution: Longest increasing subsequence
if k = 1, we need to modify the following arrays
1. [a[0], a[1], a[2], …]
if k = 2, we need to modify the following arrays
1. [a[0], a[2], a[4], …]
2. [a[1], a[3], a[5], …]
if k = 3, we need to modify the following arrays
1. [a[0], a[3], a[6], …]
2. [a[1], a[4], a[7], …]
3. [a[2], a[5], a[8], …]
…

These arrays are independent of each other, we just need to find LIS of it, # ops = len(arr) – LIS(arr).
Ans = sum(len(arri) – LIS(arri)) 1 <= i <= k

Reference: 花花酱 LeetCode 300. Longest Increasing Subsequence

Time complexity: O(k * (n/k)* log(n/k)) = O(n * log(n/k))
Space complexity: O(n/k)'''
class Solution(object):
    def kIncreasing(self, arr, k):
        """
        :type arr: List[int]
        :type k: int
        :rtype: int
        """
        def LIS(arr):
            lis = []
            for x in arr:
                if not lis or lis[-1] <= x:
                    lis.append(x)
                else:
                    lis[bisect_right(lis, x)] = x
            return len(lis)
    
        return sum(len(arr[i::k]) - LIS(arr[i::k]) for i in range(k))

# 2110. Number of Smooth Descent Periods of a Stock

# 835. Image Overlap

# 722. Remove Comments

# 1915. Number of Wonderful Substrings
'''Explanation
Use a mask to count the current prefix string.
mask & 1 means whether it has odd 'a'
mask & 2 means whether it has odd 'b'
mask & 4 means whether it has odd 'c'
...

We find the number of wonderful string with all even number of characters.
Then we flip each of bits, 10 at most, and doing this again.
This will help to find string with at most one odd number of characters.

Complexity
Time O(10n), Space O(1024)'''
class Solution(object):
    def wonderfulSubstrings(self, word):
        """
        :type word: str
        :rtype: int
        """
        count = [1] + [0] * 1024
        res = cur = 0
        for c in word:
            cur ^= 1 << (ord(c) - ord('a'))
            res += count[cur]
            res += sum(count[cur ^ (1 << i)] for i in xrange(10))
            count[cur] += 1
        return res

# 1913. Maximum Product Difference Between Two Pairs
'''Solution: Greedy
Since all the numbers are positive, we just need to find the largest two numbers as the first pair and smallest two numbers are the second pair.

Time complexity: O(nlogn) / sorting, O(n) / finding min/max elements.
Space complexity: O(1)'''
class Solution(object):
    def maxProductDifference(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        return nums[-1] * nums[-2] - nums[0] * nums[1]

# 1910. Remove All Occurrences of a Substring

# 1909. Remove One Element to Make the Array Strictly Increasing
class Solution(object):
    def canBeIncreasing(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        prev, seen = float("-inf"), False
        for i, x in enumerate(nums): 
            if prev < x: prev = x
            else: 
                if seen: return False 
                seen = True 
                if i == 1 or nums[i-2] < x: prev = x
        return True 

# 2109. Adding Spaces to a String
# Traverse input from left to right.
class Solution(object):
    def addSpaces(self, s, spaces):
        """
        :type s: str
        :type spaces: List[int]
        :rtype: str
        """
        ans = []
        j = 0
        for i, c in enumerate(s):
            if j < len(spaces) and i == spaces[j]:
                ans.append(' ')
                j += 1
            ans.append(c)
        return ''.join(ans)

# 2108. Find First Palindromic String in the Array

# 2106. Maximum Fruits Harvested After at Most K Steps

# 2105. Watering Plants II
'''
Solution: Simulation w/ Two Pointers

Simulate the watering process.

Time complexity: O(n)
Space complexity: O(1)
'''
class Solution(object):
    def minimumRefill(self, plants, capacityA, capacityB):
        """
        :type plants: List[int]
        :type capacityA: int
        :type capacityB: int
        :rtype: int
        """
        ans = 0 
        lo, hi = 0, len(plants)-1
        canA, canB = capacityA, capacityB
        while lo < hi: 
            if canA < plants[lo]: ans += 1; canA = capacityA
            canA -= plants[lo]
            if canB < plants[hi]: ans += 1; canB = capacityB
            canB -= plants[hi]
            lo, hi = lo+1, hi-1
        if lo == hi and max(canA, canB) < plants[lo]: ans += 1
        return ans 

# 2104. Sum of Subarray Ranges
'''Solution 0, Brute Force => TLE
Time O(n^3)
Space O(1)


Solution 1, Two Loops Solution
Time O(n^2)
Space O(1)
'''
class Solution(object):
    def subArrayRanges(self, A):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        n = len(A)
        for i in xrange(n):
            l,r = A[i],A[i]
            for j in xrange(i, n):
                l = min(l, A[j])
                r = max(r, A[j])
                res += r - l
        return res
'''Solution 2, O(n) Stack Solution
Follow the explanation in 907. Sum of Subarray Minimums

Intuition
res = sum(A[i] * f(i))
where f(i) is the number of subarrays,
in which A[i] is the minimum.

To get f(i), we need to find out:
left[i], the length of strict bigger numbers on the left of A[i],
right[i], the length of bigger numbers on the right of A[i].

Then,
left[i] + 1 equals to
the number of subarray ending with A[i],
and A[i] is single minimum.

right[i] + 1 equals to
the number of subarray starting with A[i],
and A[i] is the first minimum.

Finally f(i) = (left[i] + 1) * (right[i] + 1)

For [3,1,2,4] as example:
left + 1 = [1,2,1,1]
right + 1 = [1,3,2,1]
f = [1,6,2,1]
res = 3 * 1 + 1 * 6 + 2 * 2 + 4 * 1 = 17

Explanation
To calculate left[i] and right[i],
we use two increasing stacks.

It will be easy if you can refer to this problem and my post:
901. Online Stock Span
I copy some of my codes from this solution.

Complexity
All elements will be pushed twice and popped at most twice
Time O(n)
Space O(n)'''        
class Solution(object):
    def subArrayRanges(self, A0):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        inf = float('inf')
        A = [-inf] + A0 + [-inf]
        s = []
        for i, x in enumerate(A):
            while s and A[s[-1]] > x:
                j = s.pop()
                k = s[-1]
                res -= A[j] * (i - j) * (j - k)
            s.append(i)
            
        A = [inf] + A0 + [inf]
        s = []
        for i, x in enumerate(A):
            while s and A[s[-1]] < x:
                j = s.pop()
                k = s[-1]
                res += A[j] * (i - j) * (j - k)
            s.append(i)
        return res

# 2103. Rings and Rods

# 2102. Sequentially Ordinal Rank Tracker

# 2101. Detonate the Maximum Bombs

# 2100. Find Good Days to Rob the Bank

# 2099. Find Subsequence of Length K With the Largest Sum
# Method 3: Quick Select
# Time: average O(n), worst O(n ^ 2), space: O(n).
class Solution(object):
    def maxSubsequence(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # Divide index[lo...hi] into two parts: larger and less than 
        # the pivot; Then return the position of the pivot;
        def quickSelect(lo, hi) :
            pivot = index[lo]
            while lo < hi:
                while lo < hi and nums[index[hi]] <= nums[pivot]:
                    hi -= 1
                index[lo] = index[hi]
                while lo < hi and nums[index[lo]] >= nums[pivot]:
                    lo += 1
                index[hi] = index[lo]
            index[lo] = pivot
            return lo

        n = len(nums)
        index = list(range(n))
        
        # Use Quick Select to put the indexes of the 
        # max k items to the left of index array.
        left, right = 0, n - 1
        while left < right:
            idx = quickSelect(left, right)
            if idx < k:
                left = idx + 1
            else:
                right = idx
        
        # Count the occurrencs of the kth largest items
        # within the k largest ones.
        kth_val, freq_of_kth_val = nums[index[k - 1]], 0
        for i in index[ : k]:
            if nums[i] == kth_val:
                freq_of_kth_val += 1
                
        # Greedily copy the subsequence into output array seq.
        seq = []
        for num in nums:
            if num > kth_val or num == kth_val and freq_of_kth_val > 0:
                seq.append(num)
                if num == kth_val:
                    freq_of_kth_val -= 1
        return seq

# 219. Contains Duplicate II
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        dic = {}
        for i, v in enumerate(nums):
            if v in dic and i - dic[v] <= k:
                return True
            dic[v] = i
        return False

# 2097. Valid Arrangement of Pairs

# 2096. Step-By-Step Directions From a Binary Tree Node to Another
'''Solution: Lowest common ancestor

It’s no hard to see that the shortest path is from the start node to 
the lowest common ancestor (LCA) of (start, end), then to the end node. 
The key is to find the LCA while finding paths from root to two nodes.

We can use recursion to find/build a path from root to a target node.
The common prefix of these two paths is the path from root to the LCA 
that we need to remove from the shortest path.
e.g.
root to start “LLRLR”
root to dest “LLLR”
common prefix is “LL”, after removing, it becomes:
LCA to start “RLR”
LCA to dest “LR”
Final path becomes “UUU” + “LR” = “UUULR”

The final step is to replace the L/R with U for the start path 
since we are moving up and then concatenate with the target path.

Time complexity: O(n)
Space complexity: O(n)'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def getDirections(self, root, startValue, destValue):
        """
        :type root: Optional[TreeNode]
        :type startValue: int
        :type destValue: int
        :rtype: str
        """
        def lca(node): 
            """Return lowest common ancestor of start and dest nodes."""
            if not node or node.val in (startValue , destValue): return node 
            left, right = lca(node.left), lca(node.right)
            return node if left and right else left or right
        
        root = lca(root) # only this sub-tree matters
        
        ps = pd = ""
        stack = [(root, "")]
        while stack: 
            node, path = stack.pop()
            if node.val == startValue: ps = path 
            if node.val == destValue: pd = path
            if node.left: stack.append((node.left, path + "L"))
            if node.right: stack.append((node.right, path + "R"))
        return "U"*len(ps) + pd

# 2095. Delete the Middle Node of a Linked List

# 2094. Finding 3-Digit Even Numbers
class Solution(object):
    def findEvenNumbers(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        ans = []
        freq = Counter(digits)
        for x in range(100, 1000, 2): 
            if not Counter(int(d) for d in str(x)) - freq: ans.append(x)
        return ans
# Alternative O(N) approach        
class Solution(object):
    def findEvenNumbers(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        ans = set()
        for x, y, z in permutations(digits, 3): 
            if x != 0 and z & 1 == 0: 
                ans.add(100*x + 10*y + z) 
        return sorted(ans)

# 188. Best Time to Buy and Sell Stock IV

# 171. Excel Sheet Column Number
class Solution(object):
    def titleToNumber(self, columnTitle):
        """
        :type columnTitle: str
        :rtype: int
        """
        res = 0
        for c in columnTitle:
            res = res*26 + ord(c)-ord('A')+1
        return res

# 168. Excel Sheet Column Title

# 166. Fraction to Recurring Decimal

# 151. Reverse Words in a String

# 2092. Find All People With Secret

# 2091. Removing Minimum and Maximum From Array
'''
Explanation
Find index i of the minimum
Find index j of the maximum

To remove element A[i],
we can remove i + 1 elements from front,
or we can remove n - i elements from back.


Complexity
Time O(n)
Space O(1)'''
class Solution(object):
    def minimumDeletions(self, A):
        """
        :type nums: List[int]
        :rtype: int
        """
        i, j, n = A.index(min(A)), A.index(max(A)), len(A)
        return min(max(i + 1, j + 1), max(n - i, n - j), i + 1 + n - j, j + 1 + n - i)

# 2090. K Radius Subarray Averages

# 2089. Find Target Indices After Sorting Array

# 2088. Count Fertile Pyramids in a Land
'''Solution: DP
Let dp[i][j] be the height+1 of a Pyramid tops at i, j
dp[i][j] = min(dp[i+d][j – 1], dp[i + d][j + 1]) + 1 if dp[i-1][j] else grid[i][j]

Time complexity: O(mn)
Space complexity: O(mn)

'''
class Solution:
    def countPyramids(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        ans = 0

        @cache
        def dp(i, j, d):
            if grid[i][j] and 0 <= i + d < m and 0 < j < n - 1 and grid[i + d][j]:
                return min(dp(i + d, j - 1, d), dp(i + d, j + 1, d)) + 1
            return grid[i][j]

        for i, j in product(range(m), range(n)):
            ans += max(0, dp(i, j, 1) - 1)
            ans += max(0, dp(i, j, -1) - 1)

        return ans
        
# 2087. Minimum Cost Homecoming of a Robot in a Grid

# 2086. Minimum Number of Buckets Required to Collect Rainwater from Houses
'''Explanation
If s == 'H', return -1
If s starts with HH', return -1
If s ends with HH', return -1
If s has 'HHH', return -1

Each house H needs one bucket,
that's s.count('H')
Each 'H.H' can save one bucket by sharing one in the middle,
that's s.count('H.H') (greedy count without overlap)
So return s.count('H') - s.count('H.H')


Key Point
I'm not greedy,
Python count is greedy for me,
Java replace is greedy for me.


Complexity
Time O(n)
Space O(1)'''
class Solution(object):
    def minimumBuckets(self, s):
        """
        :type street: str
        :rtype: int
        """
        return -1 if 'HHH' in s or s[:2] == 'HH' or s[-2:] == 'HH' or s == 'H' else s.count('H') - s.count('H.H')
# Regax soluiton
class Solution(object):
    def minimumBuckets(self, s):
        """
        :type street: str
        :rtype: int
        """
        return -1 if search('(^|H)H(H|$)', s) else s.count('H') - s.count('H.H')

# 2085. Count Common Words With One Occurrence

# 137. Single Number II

# 114. Flatten Binary Tree to Linked List
'''Solution 1: Recursion
Time complexity: O(n)
Space complexity: O(|height|)

'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        """
        Do not return anything, modify root in-place instead.
        """
        def solve(root):
            if not root: return None, None
            if not root.left and not root.right: return root, root
            l_head = l_tail = r_head = r_tail = None
            if root.left:
                l_head, l_tail = solve(root.left)
            if root.right:
                r_head, r_tail = solve(root.right)
            root.left = None
            root.right = l_head or r_head
            if l_tail:
                l_tail.right = r_head
            return root, r_tail or l_tail
        return solve(root)[0]

'''Solution 2: Unfolding

Time complexity: O(n)
Space complexity: O(1)'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        while root:
            if root.left:
                prev = root.left
                while prev.right: prev = prev.right
                prev.right = root.right
                root.right = root.left
                root.left = None
            root = root.right

# 109. Convert Sorted List to Binary Search Tree

# 41. First Missing Positive
# O(nlgn) time
class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        res = 1
        for num in nums:
            if num == res:
                res += 1
        return res
# O(n) time
class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for i in xrange(len(nums)):
            while 0 <= nums[i]-1 < len(nums) and nums[nums[i]-1] != nums[i]:
                tmp = nums[i]-1
                nums[i], nums[tmp] = nums[tmp], nums[i]
        for i in xrange(len(nums)):
            if nums[i] != i+1:
                return i+1
        return len(nums)+1

# 30. Substring with Concatenation of All Words

# 2007. Find Original Array From Doubled Array
''' Match from the Smallest or Biggest

Intuition
Copy a solution from 954. Array of Doubled Pairs

If you have any questions, feel free to ask.
If you like solution and explanations, please Upvote!

Let's see a simple case
Assume all interger are positive, for example [2,4,4,8].
We have one x = 2, we need to match it with one 2x = 4.
Then one 4 is gone, we have the other x = 4.
We need to match it with one 2x = 8.
Finaly no number left.

Why we start from 2?
Because it's the smallest and we no there is no x/2 left.
So we know we need to find 2x

Explanation
Count all numbers.
Loop all numbers on the order of its absolute.
We have counter[x] of x, so we need the same amount of 2x wo match them.
If c[x] > c[2 * x], then we return []
If c[x] <= c[2 * x], then we repeatly do c[2 * x]-- and append x to result res
Don't worry about 0, it doesn't fit the logic above but it won't break our algorithme.

In case count[0] is odd, it won't get matched in the end.

In case count[0] is even, we still have c[0] <= c[2 * 0].
And we still need to check all other numbers.'''
class Solution(object):
    def findOriginalArray(self, A):
        """
        :type changed: List[int]
        :rtype: List[int]
        """
        c = collections.Counter(A)
        if c[0] % 2:
            return []
        for x in sorted(c):
            if c[x] > c[2 * x]:
                return []
            c[2 * x] -= c[x] if x else c[x] / 2
        return list(c.elements())

# 2008. Maximum Earnings From Taxi
'''DP solution

Explanation
Sort A, solve it like Knapsack dp.


Complexity
Time O(n + klogk), k = A.length
Space O(n)'''
class Solution(object):
    def maxTaxiEarnings(self, n, A):
        """
        :type n: int
        :type rides: List[List[int]]
        :rtype: int
        """
        dp = [0] * (n + 1)
        A.sort()
        for i in xrange(n - 1, -1, -1):
            dp[i] = dp[i + 1]
            while A and i == A[-1][0]:
                s, e, t = A.pop()
                dp[i] = max(dp[i], dp[e] + e - s + t)
        return dp[0]

# in general order 
class Solution(object):
    def maxTaxiEarnings(self, n, rides):
        """
        :type n: int
        :type rides: List[List[int]]
        :rtype: int
        """
        endAt = defaultdict(list)
        for s, e, t in rides:
            endAt[e].append((s, e - s + t))
        # 
        dp = [0] * (n + 1)
        for i in range(1, n+1):
            dp[i] = dp[i-1]
            for s, t in endAt[i]:
                dp[i] = max(dp[i], t + dp[s])
        # 
        return dp[-1]

# 2009. Minimum Number of Operations to Make Array Continuous

# 2081. Sum of k-Mirror Numbers
# Solution: Generate palindromes in base-k.
class Solution:
    def kMirror(self, k: int, n: int) -> int:
        def getNext(x: str) -> str:
          s = list(x)
          l = len(s)    
          for i in range(l # 2, l):
            if int(s[i]) + 1 >= k: continue
            s[i] = s[~i] = str(int(s[i]) + 1)
            for j in range(l # 2, i): s[j] = s[~j] = "0"
            return "".join(s)
          return "1" + "0" * (l - 1) + "1"
        ans = 0
        x = "0"
        for _ in range(n):
          while True:
            x = getNext(x)
            val = int(x, k)
            if str(val) == str(val)[::-1]: break
          ans += val
        return ans

# 2080. Range Frequency Queries
'''Binary Search

Explanation
init: Store the index for the same value.
query: Binary search the left and right in the stored indices.

Time O(qlogn)
Space O(n)'''
class RangeFreqQuery(object):

    def __init__(self, arr):
        """
        :type arr: List[int]
        """
        self.count = collections.defaultdict(list)
        for i, a in enumerate(arr):
            self.count[a].append(i)
        

    def query(self, left, right, value):
        """
        :type left: int
        :type right: int
        :type value: int
        :rtype: int
        """
        i = bisect.bisect(self.count[value], left - 1)
        j = bisect.bisect(self.count[value], right)
        return j - i


# Your RangeFreqQuery object will be instantiated and called as such:
# obj = RangeFreqQuery(arr)
# param_1 = obj.query(left,right,value)

# 2079. Watering Plants

# 2050. Parallel Courses III
'''Solution: Topological Sorting
Time complexity: O(V+E)
Space complexity: O(V+E)
'''
class Solution:
    def minimumTime(self, n: int, relations: List[List[int]], time: List[int]) -> int:
        g = [[] for _ in range(n)]
        for u, v in relations: g[u - 1].append(v - 1)
        @cache
        def dfs(u: int) -> int:      
          return max([dfs(v) for v in g[u]] + [0]) + time[u]
        return max(dfs(u) for u in range(n))

# 2003. Smallest Missing Genetic Value in Each Subtree

# 2002. Maximum Product of the Length of Two Palindromic Subsequences

# 2001. Number of Pairs of Interchangeable Rectangles

# 2000. Reverse Prefix of Word

# 2065. Maximum Path Quality of a Graph
#  Concise DFS
class Solution(object):
    def maximalPathQuality(self, A, edges, maxTime):
        """
        :type values: List[int]
        :type edges: List[List[int]]
        :type maxTime: int
        :rtype: int
        """
        G = collections.defaultdict(dict)
        for i, j, t in edges:
            G[i][j] = G[j][i] = t

        def dfs(i, seen, time):
            res = sum(A[j] for j in seen) if i == 0 else 0
            for j in G[i]:
                if time >= G[i][j]:
                    res = max(res, dfs(j, seen | {j}, time - G[i][j]))
            return res

        return dfs(0, {0}, maxTime)

# 2068. Check Whether Two Strings are Almost Equivalent

# 2069. Walking Robot Simulation II

# 2070. Most Beautiful Item for Each Query
class Solution(object):
    def maximumBeauty(self, A, queries):
        """
        :type items: List[List[int]]
        :type queries: List[int]
        :rtype: List[int]
        """
        A = sorted(A + [[0, 0]])
        for i in xrange(len(A) - 1):
            A[i + 1][1] = max(A[i][1], A[i + 1][1])
        return [A[bisect.bisect(A, [q + 1]) - 1][1] for q in queries]

# 2071. Maximum Number of Tasks You Can Assign

# 2073. Time Needed to Buy Tickets

# 2074. Reverse Nodes in Even Length Groups

# 2075. Decode the Slanted Ciphertext

# 2076. Process Restricted Friend Requests

# 2047. Number of Valid Words in a Sentence
'''Solution 2: Regex
Time complexity: O(n^2)?
Space complexity: O(1)
'''
class Solution:
    def countValidWords(self, sentence: str) -> int:
        ans = 0
        for word in sentence.split():
            if word.strip() and re.fullmatch('^([a-z]+(-?[a-z]+)?)?[\.,!]?$', word.strip()):
                ans += 1
        return ans

# 1899. Merge Triplets to Form Target Triplet

# 1900. The Earliest and Latest Rounds Where Players Compete

# 1903. Largest Odd Number in String

# 1904. The Number of Full Rounds You Have Played

# 1897. Redistribute Characters to Make All Strings Equal
'''Solution: Hashtable
Count the frequency of each character, it must be a multiplier of n such that we can evenly distribute it to all the words.
e.g. n = 3, a = 9, b = 6, c = 3, each word will be “aaabbc”.

Time complexity: O(n)
Space complexity: O(1)'''
class Solution(object):
    def makeEqual(self, words):
        """
        :type words: List[str]
        :rtype: bool
        """
        return all(c % len(words) == 0 for c in Counter(''.join(words)).values())

# 1896. Minimum Cost to Change the Final Value of Expression

# 1895. Largest Magic Square

# 1894. Find the Student that Will Replace the Chalk

# 1893. Check if All the Integers in a Range Are Covered
'''Explanation
all values in range(left, right + 1),
should be in any one interval (l, r).


Complexity
Time O((right - left) * n),
where n = ranges.length
Space O(1)'''
class Solution(object):
    def isCovered(self, ranges, left, right):
        """
        :type ranges: List[List[int]]
        :type left: int
        :type right: int
        :rtype: bool
        """
        return all(any(l <= i <= r for l, r in ranges) for i in xrange(left, right + 1))

# 1887. Reduction Operations to Make the Array Elements Equal

# 1888. Minimum Number of Flips to Make the Binary String Alternating

# 1889. Minimum Space Wasted From Packaging

# 1880. Check if Word Equals Summation of Two Words

# 1879. Minimum XOR Sum of Two Arrays

# 1878. Get Biggest Three Rhombus Sums in a Grid

# 1877. Minimize Maximum Pair Sum in Array

# 1876. Substrings of Size Three with Distinct Characters
'''Solution: Brute Force w/ (Hash)Set
Time complexity: O(n)
Space complexity: O(1)'''
class Solution(object):
    def countGoodSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        return sum(len(set(s[i:i + 3])) == 3 for i in range(len(s) - 2))

# 1872. Stone Game VIII

# 1871. Jump Game VII
'''One Pass DP

Explanation
dp[i] = true if we can reach s[i].
pre means the number of previous position that we can jump from.

Complexity
Time O(n)
Space O(n)'''
class Solution(object):
    def canReach(self, s, minJump, maxJump):
        """
        :type s: str
        :type minJump: int
        :type maxJump: int
        :rtype: bool
        """
        dp = [c == '0' for c in s]
        pre = 0
        for i in xrange(1, len(s)):
            if i >= minJump: pre += dp[i - minJump]
            if i > maxJump: pre -= dp[i - maxJump - 1]
            dp[i] &= pre > 0
        return dp[-1]

# 1869. Longer Contiguous Segments of Ones than Zeros

# 1866. Number of Ways to Rearrange Sticks With K Sticks Visible
'''Solution: DP
dp(n, k) = dp(n – 1, k – 1) + (n-1) * dp(n-1, k)

Time complexity: O(n*k)
Space complexity: O(n*k) -> O(k)'''
class Solution:
    @lru_cache(maxsize=None)
    def rearrangeSticks(self, n: int, k: int) -> int:
        if k == 0: return 0
        if k == n or n <= 2: return 1
        return (self.rearrangeSticks(n - 1, k - 1) + 
                (n - 1) * self.rearrangeSticks(n - 1, k)) % (10**9 + 7)

# 1865. Finding Pairs With a Certain Sum
'''Solution: HashTable
Note nums1 and nums2 are unbalanced. 
Brute force method will take O(m*n) = O(103*105) = O(108) 
for each count call which will TLE. 
We could use a hashtable to store the counts of elements from nums2, 
and only iterate over nums1 to reduce the time complexity.

Time complexity:

init: O(m) + O(n)
add: O(1)
count: O(m)

Total time is less than O(106)

Space complexity: O(m + n)'''
class FindSumPairs:

    def __init__(self, nums1: List[int], nums2: List[int]):
        self.nums1 = nums1
        self.nums2 = nums2
        self.freq = Counter(nums2)

    def add(self, index: int, val: int) -> None:
        self.freq.subtract((self.nums2[index],))
        self.nums2[index] += val
        self.freq.update((self.nums2[index],))

    def count(self, tot: int) -> int:
        return sum(self.freq[tot - a] for a in self.nums1)


# Your FindSumPairs object will be instantiated and called as such:
# obj = FindSumPairs(nums1, nums2)
# obj.add(index,val)
# param_2 = obj.count(tot)

# 1864. Minimum Number of Swaps to Make the Binary String Alternating

# 1863. Sum of All Subset XOR Totals
'''Solution 1: Brute Force
Use an array A to store all the xor subsets, for a given number x
A = A + [x ^ a for a in A]

Time complexity: O(2n)
Space complexity: O(2n)'''
class Solution:
    def subsetXORSum(self, nums: List[int]) -> int:
        xors = [0]
        for x in nums:
          xors += [xor ^ x for xor in xors]    
        return sum(xors)

# 1860. Incremental Memory Leak
'''Solution: Simulation
Time complexity: O(max(memory1, memory2)0.5)
Space complexity: O(1)

'''
class Solution:
    def memLeak(self, memory1: int, memory2: int) -> List[int]:
        for i in range(1, 2**30):
          if max(memory1, memory2) < i:
            return [i, memory1, memory2]
          elif memory1 >= memory2:
            memory1 -= i
          else:
            memory2 -= i
        return None

# 1859. Sorting the Sentence
'''Solution: String
Time complexity: O(n)
Space complexity: O(n)'''
class Solution:
    def sortSentence(self, s: str) -> str:
        p = [(w[:-1], int(w[-1])) for w in s.split()]
        p.sort(key=lambda x : x[1])
        return " ".join((w for w, _ in p))

# 1857. Largest Color Value in a Directed Graph
'''Solution: Topological Sorting
freq[n][c] := max freq of color c after visiting node n.

Time complexity: O(n)
Space complexity: O(n*26)

'''
class Solution:
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        INF = 1e9
        n = len(colors)
        g = [[] for _ in range(n)]
        for u, v in edges:
          g[u].append(v)    
        visited = [0] * n
        freq = [[0] * 26 for _ in range(n)]
        def dfs(u: int) -> int:
          idx = ord(colors[u]) - ord('a')
          if not visited[u]:
            visited[u] = 1 # visiting
            for v in g[u]:
              if (dfs(v) == INF):
                return INF
              for c in range(26):
                freq[u][c] = max(freq[u][c], freq[v][c])
            freq[u][idx] += 1
            visited[u] = 2 # done
          return freq[u][idx] if visited[u] == 2 else INF
        ans = 0
        for u in range(n):
          ans = max(ans, dfs(u))
          if ans == INF: break
        return -1 if ans == INF else ans

# 1854. Maximum Population Year

# 1851. Minimum Interval to Include Each Query
'''Priority Queue Solution

Explanation
Sort queries and intervals.
Iterate queries from small to big,
and find out all open intervals [l, r],
and we add them to a priority queue.
Also, we need to remove all closed interval from the queue.

In the priority, we use
[interval size, interval end] = [r-l+1, r] as the key.

The head of the queue is the smallest interval we want to return for each query.


Complexity
Time O(nlogn + qlogq)
Space O(n+q)
where q = queries.size()
'''
class Solution(object):
    def minInterval(self, A, queries):
        """
        :type intervals: List[List[int]]
        :type queries: List[int]
        :rtype: List[int]
        """
        A = sorted(A)[::-1]
        h = []
        res = {}
        for q in sorted(queries):
            while A and A[-1][0] <= q:
                i, j = A.pop()
                if j >= q:
                    heapq.heappush(h, [j - i + 1, j])
            while h and h[0][1] < q:
                heapq.heappop(h)
            res[q] = h[0][0] if h else -1
        return [res[q] for q in queries]

# 1850. Minimum Adjacent Swaps to Reach the Kth Smallest Number

# 1849. Splitting a String Into Descending Consecutive Values

# 1848. Minimum Distance to the Target Element

# 1847. Closest Room

# 1846. Maximum Element After Decreasing and Rearranging

# 1844. Replace All Digits with Characters
''' Straight Forward

Complexity
Time O(n)
Space O(n)'''
class Solution(object):
    def replaceDigits(self, s):
        """
        :type s: str
        :rtype: str
        """
        return ''.join(chr(ord(s[i-1]) + int(s[i])) if i % 2 else s[i] for i in xrange(len(s)))

# 1840. Maximum Building Height

# 1839. Longest Substring Of All Vowels in Order

# 1837. Sum of Digits in Base K

# 1835. Find XOR Sum of All Pairs Bitwise AND

# 1834. Single-Threaded CPU

# 1833. Maximum Ice Cream Bars

# 1832. Check if the Sentence Is Pangram
class Solution(object):
    def checkIfPangram(self, sentence):
        """
        :type sentence: str
        :rtype: bool
        """
        return len(set(sentence)) == 26

# 1830. Minimum Number of Operations to Make String Sorted

# 1829. Maximum XOR for Each Query

# 1828. Queries on Number of Points Inside a Circle

# 1827. Minimum Operations to Make the Array Increasing

# 1825. Finding MK Average

# 1824. Minimum Sideway Jumps

# 1819. Number of Different Subsequences GCDs

# 1817. Finding the Users Active Minutes

# 1816. Truncate Sentence
'''Solution:
Time complexity: O(n)
Space complexity: O(n)'''
class Solution(object):
    def truncateSentence(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        return " ".join(s.split()[:k])

# 1815. Maximum Number of Groups Getting Fresh Donuts

# 1814. Count Nice Pairs in an Array
'''Straight Forward

Explanation
A[i] + rev(A[j]) == A[j] + rev(A[i])
A[i] - rev(A[i]) == A[j] - rev(A[j])
B[i] = A[i] - rev(A[i])

Then it becomes an easy question that,
how many pairs in B with B[i] == B[j]


Complexity
Time O(nloga)
Space O(n)'''
class Solution(object):
    def countNicePairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        count = collections.Counter()
        for a in nums:
            b = int(str(a)[::-1])
            res += count[a - b]
            count[a - b] += 1
        return res % (10**9 + 7)

# 1813. Sentence Similarity III
'''Solution: Dequeue / Common Prefix + Suffix
Break sequences to words, store them in two deques. 
Pop the common prefix and suffix. 
At least one of the deque should be empty.

Time complexity: O(m+n)
Space complexity: O(m+n)

'''
class Solution:
    def areSentencesSimilar(self, sentence1: str, sentence2: str) -> bool:
        w1 = deque(sentence1.split())
        w2 = deque(sentence2.split())
        while w1 and w2 and w1[0] == w2[0]:
          w1.popleft(), w2.popleft()
        while w1 and w2 and w1[-1] == w2[-1]:
          w1.pop(), w2.pop()
        return len(w1) * len(w2) == 0

# 1812. Determine Color of a Chessboard Square

# 1808. Maximize Number of Nice Divisors

# 1807. Evaluate the Bracket Pairs of a String\

# 1806. Minimum Number of Operations to Reinitialize a Permutation

# 1805. Number of Different Integers in a String
class Solution(object):
    def numDifferentIntegers(self, word):
        """
        :type word: str
        :rtype: int
        """
        s = ''.join(c if c.isdigit() else ' ' for c in word)
        return len(set(map(int, s.split())))

class Solution(object):
    def numDifferentIntegers(self, word):
        """
        :type word: str
        :rtype: int
        """
        return len(set(map(int, re.findall(r'\d+', word))))

# 1801. Number of Orders in the Backlog
'''Priority Queue

Complexity
Time O(nlogn)
Space O(n)'''
class Solution(object):
    def getNumberOfBacklogOrders(self, orders):
        """
        :type orders: List[List[int]]
        :rtype: int
        """
        sell, buy = [], []
        for p, a, t in orders:
            if t == 0:
                heapq.heappush(buy, [-p, a])
            else:
                heapq.heappush(sell, [p, a])
            while sell and buy and sell[0][0] <= -buy[0][0]:
                k = min(buy[0][1], sell[0][1])
                buy[0][1] -= k
                sell[0][1] -= k
                if buy[0][1] == 0: heapq.heappop(buy)
                if sell[0][1] == 0: heapq.heappop(sell)
        return sum(a for p, a in buy + sell) % (10**9 + 7)

# 1800. Maximum Ascending Subarray Sum

# 1799. Maximize Score After N Operations

# 1798. Maximum Number of Consecutive Values You Can Make

# 1796. Second Largest Digit in a String

# 1793. Maximum Score of a Good Subarray

# 1792. Maximum Average Pass Ratio
'''Solution: Greedy + Heap

Sort by the ratio increase potential (p + 1) / (t + 1) - p / t.

Time complexity: O((m+n)logn)
Space complexity: O(n)'''
class Solution:
    def maxAverageRatio(self, classes: List[List[int]], extraStudents: int) -> float:
        def ratio(i, delta=0):
          return (classes[i][0] + delta) / (classes[i][1] + delta)
        q = []
        for i, c in enumerate(classes):
          heapq.heappush(q, (-(ratio(i, 1) - ratio(i)), i))
        for _ in range(extraStudents):
          _, i = heapq.heappop(q)
          classes[i][0] += 1
          classes[i][1] += 1
          heapq.heappush(q, (-(ratio(i, 1) - ratio(i)), i))
        return mean(ratio(i) for i, _ in enumerate(classes))

# 1791. Find Center of Star Graph
'''Since the center node must appear in each edge, 
we just need to find the mode of edges[0] + edges[1]

Time complexity: O(1)
Space complexity: O(1)'''
class Solution:
    def findCenter(self, edges: List[List[int]]) -> int:
        return mode(edges[0] + edges[1])

# 1727. Largest Submatrix With Rearrangements

# 1732. Find the Highest Altitude

# 1733. Minimum Number of People to Teach

# 1734. Decode XORed Permutation

# 1736. Latest Time by Replacing Hidden Digits

# 1737. Change Minimum Characters to Satisfy One of Three Conditions
'''Clean Solution

Explanation
Count the frequcy of each character in a and b.
Find the most common characters most_common = max((c1 + c2).values()),
this help meet the condition 3 with m + n - most_common.

The we calculate the accumulate prefix sum of count.
This help finding the number of smaller characters in O(1) time.

Enumerate the character i a,b,c...x,y,
To meet condition 1,
which is a < b,
we need (m - c1[i]) + c2[i]

To meet condition 2,
which is a > b,
we need n - c2[i] + c1[i]


Complexity
Time O(m + n)
Space O(26)'''
class Solution(object):
    def minCharacters(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: int
        """
        m, n = len(a), len(b)
        c1 = Counter(ord(c) - 97 for c in a)
        c2 = Counter(ord(c) - 97 for c in b)
        res = m + n - max((c1 + c2).values()) # condition 3
        for i in range(25):
            c1[i + 1] += c1[i]
            c2[i + 1] += c2[i]
            res = min(res, m - c1[i] + c2[i]) # condition 1
            res = min(res, n - c2[i] + c1[i]) # condition 2
        return res

# 1738. Find Kth Largest XOR Coordinate Value

# 1739. Building Boxes

# 1742. Maximum Number of Balls in a Box
'''Solution: Hashtable and base-10
Max sum will be 9+9+9+9+9 = 45

Time complexity: O((hi-lo) * log(hi))
Space complexity: O(1)'''
class Solution(object):
    def countBalls(self, lowLimit, highLimit):
        """
        :type lowLimit: int
        :type highLimit: int
        :rtype: int
        """
        balls = defaultdict(int)
        ans = 0
        for x in range(lowLimit, highLimit + 1):
          s = sum(int(d) for d in str(x))
          balls[s] += 1
          ans = max(ans, balls[s])
        return ans

# 1743. Restore the Array From Adjacent Pairs

# 1744. Can You Eat Your Favorite Candy on Your Favorite Day?

# 1745. Palindrome Partitioning IV

# 1748. Sum of Unique Elements

# 1749. Maximum Absolute Sum of Any Subarray

# 1750. Minimum Length of String After Deleting Similar Ends

# 1751. Maximum Number of Events That Can Be Attended II
'''DP

Explanation
For each meeting,
find the maximum value we can get before this meeting starts.
Repeatly doing this K times.


Complexity
Time O(knlogn), can be improved to O(nk) like Knapsack problem
Space O(n)'''
class Solution(object):
    def maxValue(self, events, k):
        """
        :type events: List[List[int]]
        :type k: int
        :rtype: int
        """
        events.sort(key=lambda sev: sev[1])
        dp, dp2 = [[0, 0]], [[0, 0]]
        for k0 in xrange(k):
            for s, e, v in events:
                i = bisect.bisect(dp, [s]) - 1
                if dp[i][1] + v > dp2[-1][1]:
                    dp2.append([e, dp[i][1] + v])
            dp, dp2 = dp2, [[0, 0]]
        return dp[-1][-1]

# 1752. Check if Array Is Sorted and Rotated
''' Easy and Concise

Explanation
Compare all neignbour elements (a,b) in A,
the case of a > b can happen at most once.

Note that the first element and the last element are also connected.

If all a <= b, A is already sorted.
If all a <= b but only one a > b,
we can rotate and make b the first element.
Other case, return false.


Complexity
Time O(n)
Space O(1)
''''
class Solution(object):
    def check(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        return sum(a > b for a, b in zip(nums, nums[1:] + nums[:1])) <= 1

# 1753. Maximum Score From Removing Stones

# 1754. Largest Merge Of Two Strings
''' Easy Greedy

Explanation
Just compare the string s1 and s2,
if s1 >= s2, take from s1
if s1 < s2, take from s2

it makes sense once you come up with it.


Complexity
Feel like it's
Time O(m^2+n^2)
Space O(m^2+n^2)'''
class Solution(object):
    def largestMerge(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: str
        """
        if word1 >= word2 > '':
            return word1[0] + self.largestMerge(word1[1:], word2)
        if word2 >= word1 > '':
            return word2[0] + self.largestMerge(word1, word2[1:])
        return word1 + word2

# 1755. Closest Subsequence Sum

# 1758. Minimum Changes To Make Alternating Binary String

# 1759. Count Number of Homogenous Substrings

# 1700. Number of Students Unable to Eat Lunch

# 1701. Average Waiting Time
'''Solution: Simulation
When a customer arrives, if the arrival time is greater than current, 
then advance the clock to arrival time. 
Advance the clock by cooking time. Waiting time = current time - arrival time.

Time complexity: O(n)
Space complexity: O(1)'''
class Solution:
    def averageWaitingTime(self, customers: List[List[int]]) -> float:
        cur = 0
        waiting = 0
        for arrival, time in customers:
          cur = max(cur, arrival) + time
          waiting += cur - arrival
        return waiting / len(customers)

# 1702. Maximum Binary String After Change
'''Solution with Explanation
121
lee215's avatar
lee215
168265
Last Edit: January 8, 2021 2:33 PM

4.8K VIEWS

Explanation
We don't need touch the starting 1s, they are already good.

For the rest part,
we continually take operation 2,
making the string like 00...00011...11

Then we continually take operation 1,
making the string like 11...11011...11.


Complexity
Time O(n)
Space O(n)'''
class Solution(object):
    def maximumBinaryString(self, binary):
        """
        :type binary: str
        :rtype: str
        """
        if '0' not in binary: return binary
        k, n = binary.count('1', binary.find('0')), len(binary)
        return '1' * (n - k - 1) + '0' + '1' * k

# 1703. Minimum Adjacent Swaps for K Consecutive Ones
'''Solution: Prefix Sum + Sliding Window
Time complexity: O(n)
Space complexity: O(n)

We only care positions of 1s, we can move one element from position x to y 
(assuming x + 1 ~ y are all zeros) in y - x steps. 
e.g. [0 0 1 0 0 0 1] => [0 0 0 0 0 1 1], 
move first 1 at position 2 to position 5, cost is 5 - 2 = 3.

Given a size k window of indices of ones, 
the optimal solution it to use the median number as center. 
We can compute the cost to form consecutive numbers:

e.g. [1 4 7 9 10] => [5 6 7 8 9] cost = (5 - 1) + (6 - 4) + (9 - 8) + (10 - 9) = 8

However, naive solution takes O(n*k) => TLE.

We can use prefix sum to compute the cost of a window in O(1) to 
reduce time complexity to O(n)

First, in order to use sliding window, 
we change the target of every number in the window to the median number.
e.g. [1 4 7 9 10] => [7 7 7 7 7] cost = (7 – 1) + (7 – 4) + (7 – 7) + (9 – 7) + (10 – 7) = (9 + 10) – (1 + 4) = right – left.
[5 6 7 8 9] => [7 7 7 7 7] takes extra 2 + 1 + 1 + 2 = 6 steps = (k / 2) * ((k + 1) / 2), 
these extra steps should be deducted from the final answer.'''
class Solution(object):
    def minMoves(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        ans = 1e10
        s = [0]    
        for i, v in enumerate(nums):
          if v: s.append(s[-1] + i)
        n = len(s)
        m1 = k # 2
        m2 = (k + 1) # 2
        for i in range(n - k):
          right = s[i + k] - s[i + m1]
          left = s[i + m2] - s[i]
          ans = min(ans, right - left)
        return ans - m1 * m2

# 1704. Determine if String Halves Are Alike
'''Solution: Counting
Time complexity: O(n)
Space complexity: O(1)'''
class Solution:
    def halvesAreAlike(self, s: str) -> bool:
        def count(s: str) -> int:
            return sum(c in 'aeiouAEIOU' for c in s)
        n = len(s)
        return count(s[:n#2]) == count(s[n#2:])

# 1705. Maximum Number of Eaten Apples

# 1711. Count Good Meals
class Solution(object):
    def countPairs(self, deliciousness):
        """
        :type deliciousness: List[int]
        :rtype: int
        """
        sums = [1<<i for i in range(22)]
        m = defaultdict(int)
        ans = 0
        for x in deliciousness:
          for t in sums:
            if t - x in m: ans += m[t - x]
          m[x] += 1
        return ans % (10**9 + 7)

# 1710. Maximum Units on a Truck

# 1716. Calculate Money in Leetcode Bank

# 1717. Maximum Score From Removing Substrings

# 1718. Construct the Lexicographically Largest Valid Sequence
class Solution(object):
    def constructDistancedSequence(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        l = n * 2 - 1
        ans = [0] * l
        def dfs(i, s):
          if i == l: return True
          if ans[i]: return dfs(i + 1, s)
          for d in range(n, 0, -1):
            j = i + (0 if d == 1 else d)
            if s & (1 << d) or j >= l or ans[j]: continue
            ans[i] = ans[j] = d
            if dfs(i + 1, s | (1 << d)): return True
            ans[i] = ans[j] = 0
          return False
        dfs(0, 0)
        return ans

# 1719. Number Of Ways To Reconstruct A Tree

# 1720. Decode XORed Array
'''Solution: Bitset
Time complexity: O(E*V)
Space complexity: O(V^2)'''

# 1721. Swapping Nodes in a Linked List

# 1722. Minimize Hamming Distance After Swap Operations

# 1723. Find Minimum Time to Finish All Jobs
'''Solution 2: Bianry search
The problem of the first solution,
is that the upper bound reduce not quick enough.
Apply binary search, to reduce the upper bound more quickly.'''
class Solution(object):
    def minimumTimeRequired(self, A, k):
        """
        :type jobs: List[int]
        :type k: int
        :rtype: int
        """
        n = len(A)
        A.sort(reverse=True) # opt 1

        def dfs(i):
            if i == n: return True # opt 3
            for j in xrange(k):
                if cap[j] >= A[i]:
                    cap[j] -= A[i]
                    if dfs(i + 1): return True
                    cap[j] += A[i]
                if cap[j] == x: break # opt 2
            return False

        # binary search
        left, right = max(A), sum(A)
        while left < right:
            x = (left + right) / 2
            cap = [x] * k
            if dfs(0):
                right = x
            else:
                left = x + 1
        return left

# 1725. Number Of Rectangles That Can Form The Largest Square

# 1726. Tuple with Same Product

# 1691. Maximum Height by Stacking Cuboids
'''
DP, Prove with Explanation

Intuition
There is something midleading here, you need to understand the differnece.
If the question is:
"You can place cuboid i on cuboid j if width[i] <= width[j] 
and length[i] <= length[j]"
that's will be difficult.

But it's
"You can place cuboid i on cuboid j if width[i] <= width[j] 
and length[i] <= length[j] and height[i] <= height[j]"
That's much easier.


Explanation
You can rearrange any cuboid's dimensions by rotating it to put it on another cuboid.
So for each cuboid, we sort its length in three dimension.

You can place cuboid i on cuboid j,
we have
width[i] <= width[j] and length[i] <= length[j] and height[i] <= height[j].

This condition will hold, after we sort each cuboid length,
that is,
small[i] <= small[j] and mid[i] <= mid[j] and big[i] <= big[j].

We apply a brute for doulbe for loop,
to compare each pair of cuboids,
check if they satifify the condition samll[i] <= small[j] 
and mid[i] <= mid[j] and big[i] <= big[j]
If so, we can place cuboid i on cuboid j.

You may concern whether area[i] <= area[j].
Don't worry, we always put the big[i] as the height,
the area (width,length) = (small[i], mid[j]),
and we have checked samll[i] <= small[j] && mid[i] <= mid[j].


Complexity
Time O(n^2)
Space O(n)'''
class Solution(object):
    def maxHeight(self, cuboids):
        """
        :type cuboids: List[List[int]]
        :rtype: int
        """
        cuboids = [[0, 0, 0]] + sorted(map(sorted,cuboids))
        dp = [0] * len(cuboids)
        for j in xrange(1, len(cuboids)):
            for i in xrange(j):
                if all(cuboids[i][k] <= cuboids[j][k] for k in xrange(3)):
                    dp[j] = max(dp[j], dp[i] + cuboids[j][2])
        return max(dp)

# 1690. Stone Game VII
'''Solution: MinMax + DP

For a sub game of stones[l~r] game(l, r), we have two choices:
Remove the left one: sum(stones[l + 1 ~ r]) – game(l + 1, r)
Remove the right one: sum(stones[l ~ r – 1]) – game(l, r – 1)
And take the best choice.

Time complexity: O(n^2)
Space complexity: O(n^2)'''
class Solution(object):
    def stoneGameVII(self, stones):
        """
        :type stones: List[int]
        :rtype: int
        """
        n = len(stones)
        s = [0] * (n + 1)
        for i in range(n): s[i + 1] = s[i] + stones[i]
        dp = [[0] * n for _ in range(n)]
        for c in range(2, n + 1):
          for l in range(0, n - c + 1):
            r = l + c - 1
            dp[l][r] = max(s[r + 1] - s[l + 1] - dp[l + 1][r],
                           s[r] - s[l] - dp[l][r - 1])
        return dp[0][n - 1]

# 1689. Partitioning Into Minimum Number Of Deci-Binary Numbers 
class Solution(object):
    def minPartitions(self, n):
        """
        :type n: str
        :rtype: int
        """
        return int(max(n))

# 1679. Max Number of K-Sum Pairs
class Solution(object):
    def maxOperations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        m = defaultdict(int)
        ans = 0
        for x in nums: m[x] += 1
        for x in nums:
          if m[x] < 1 or m[k - x] < 1 + (x + x == k): continue
          m[x] -= 1
          m[k - x] -= 1
          ans += 1
        return ans
        
# 1680. Concatenation of Consecutive Binary Numbers

# 1681. Minimum Incompatibility

# 1684. Count the Number of Consistent Strings
class Solution(object):
    def countConsistentStrings(self, allowed, words):
        """
        :type allowed: str
        :type words: List[str]
        :rtype: int
        """
        return sum(all(c in allowed for c in w) for w in words)

# 1685. Sum of Absolute Differences in a Sorted Array

# 1675. Minimize Deviation in Array
'''Explanation
For each a in A,
divide a by 2 until it is an odd.
Push divided a and its original value in to the pq.

The current max value in pq is noted as ma.
We iterate from the smallest value in pq,
Update res = min(res, ma - a),
then we check we can get a * 2.

If a is an odd, we can get a * 2,
If a < a0, which is its original value, we can also get a*2.

If we can, we push [a*2,a0] back to the pq and continue this process.


Complexity
Time O(nlogn)
Space O(n)


Solution 1: Use Priority Queue'''
class Solution(object):
    def minimumDeviation(self, A):
        """
        :type nums: List[int]
        :rtype: int
        """
        pq = []
        for a in A:
            heapq.heappush(pq, [a / (a & -a), a])
        res = float('inf')
        ma = max(a for a, a0 in pq)
        while len(pq) == len(A):
            a, a0 = heapq.heappop(pq)
            res = min(res, ma - a)
            if a % 2 == 1 or a < a0:
                ma = max(ma, a * 2)
                heapq.heappush(pq, [a * 2, a0])
        return res

# 1673. Find the Most Competitive Subsequence
'''Solution: Stack
Use a stack to track the best solution so far, 
pop if the current number is less than the top of the stack and 
there are sufficient numbers left. 
Then push the current number to the stack if not full.

Time complexity: O(n)
Space complexity: O(k)'''
class Solution(object):
    def mostCompetitive(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        ans = [None] * k
        n = len(nums)
        c = 0
        for i, x in enumerate(nums):
          while c and ans[c - 1] > x and c + n - i - 1 >= k:
            c -= 1
          if c < k: 
            ans[c] = x
            c += 1
        return ans

# 1663. Smallest String With A Given Numeric Value
'''Solution: Greedy, Fill in reverse order
Fill the entire string with 'a', k-=n, 
then fill in reverse order, replace 'a' with 'z' until not enough k left.

Time complexity: O(n)
Space complexity: O(n)'''
class Solution:
    def getSmallestString(self, n: int, k: int) -> str:
        ans = ['a'] * n
        k -= n
        i = n - 1
        while k:
          d = min(k, 25)
          ans[i] = chr(ord(ans[i]) + d)
          k -= d
          i -= 1
        return ''.join(ans)

# 1656. Design an Ordered Stream
'''Solution: Straight Forward
Time complexity: O(n) in total
Space complexity: O(n)

'''
class OrderedStream:

    def __init__(self, n: int):
        self.data = [None] * (n + 1)
        self.ptr = 1

    def insert(self, idKey: int, value: str) -> List[str]:
        self.data[idKey] = value
        if idKey == self.ptr:
          while self.ptr < len(self.data) and self.data[self.ptr]:
            self.ptr += 1
          return self.data[idKey:self.ptr]
        return []

# 1657. Determine if Two Strings Are Close
'''Solution: Hashtable
Two strings are close:
1. Have the same length, ccabbb => 6 == aabccc => 6
2. Have the same char set, ccabbb => (a, b, c) == aabccc => (a, b, c)
3. Have the same sorted char counts ccabbb => (1, 2, 3) == aabccc => (1, 2, 3)

Time complexity: O(n)
Space complexity: O(1)'''
class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        c1, c2 = Counter(word1), Counter(word2)
        return all([len(word1) == len(word2), 
                    c1.keys() == c2.keys(),
                    sorted(c1.values()) == sorted(c2.values())])

# 1627. Graph Connectivity With Threshold
'''Solution: Union Find
For x, merge 2x, 3x, 4x, ..,
If a number is already “merged”, skip it.

Time complexity: O(nlogn? + queries)?
Space complexity: O(n)'''
class Solution:
    def areConnected(self, n: int, threshold: int, queries: List[List[int]]) -> List[bool]:
        if threshold == 0: return [True] * len(queries)
    
        ds = list(range(n + 1))
        def find(x: int) -> int:
          if x != ds[x]: ds[x] = find(ds[x])
          return ds[x]

        for x in range(threshold + 1, n + 1):
          if ds[x] == x:
            for y in range(2 * x, n + 1, x):
              ds[max(find(x), find(y))] = min(find(x), find(y))

        return [find(x) == find(y) for x, y in queries]

# 1616. Split Two Strings to Make Palindrome
'''Greedy Solution, O(1) Space

Explanation
Greedily take the a_suffix and b_prefix as long as they are palindrome,
that is, a_suffix = reversed(b_prefix).

The the middle part of a is s1,
The the middle part of b is s2.

If either s1 or s2 is palindrome, then return true.

Then we do the same thing for b_suffix and a_prefix


Solution 1:
Time O(N), Space O(N)'''
class Solution(object):
    def checkPalindromeFormation(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: bool
        """
        i, j = 0, len(a) - 1
        while i < j and a[i] == b[j]:
            i, j = i + 1, j - 1
        s1, s2 = a[i:j + 1], b[i:j + 1]

        i, j = 0, len(a) - 1
        while i < j and b[i] == a[j]:
            i, j = i + 1, j - 1
        s3, s4 = a[i:j + 1], b[i:j + 1]

        return any(s == s[::-1] for s in (s1,s2,s3,s4))

#  1609. Even Odd Tree
'''Solution 2: BFS
Time complexity: O(n)
Space complexity: O(n)'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isEvenOddTree(self, root: Optional[TreeNode]) -> bool:
        q = collections.deque([root])
        odd = 1
        while q:
          prev = 0 if odd else 10**7
          for _ in range(len(q)):
            root = q.popleft()
            if not root: continue
            comp = int.__le__ if odd else int.__ge__        
            if root.val % 2 != odd or comp(root.val, prev): return False        
            prev = root.val
            q += (root.left, root.right)        
          odd = 1 - odd
        return True

'''Solution 1: DFS
Time complexity: O(n)
Space complexity: O(n)'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isEvenOddTree(self, root: Optional[TreeNode]) -> bool:
        vals = {}
        def dfs(root: TreeNode, d: int) -> bool:
          if not root: return True
          if d not in vals: vals[d] = 0 if d % 2 == 0 else 10**7
          comp = int.__ge__ if d % 2 else int.__le__      
          if root.val % 2 == d % 2 or comp(root.val, vals[d]): return False      
          vals[d] = root.val
          return dfs(root.left, d + 1) and dfs(root.right, d + 1)
        return dfs(root, 0)

# 1600. Throne Inheritance
'''Solution: HashTable + DFS
Record :
1. mapping from parent to children (ordered)
2. who has dead

Time complexity: getInheritanceOrder O(n), other O(1)
Space complexity: O(n)'''
class ThroneInheritance:

    def __init__(self, kingName: str):
        self.kingName = kingName
        self.family = defaultdict(list)    
        self.dead = set()

    def birth(self, parentName: str, childName: str) -> None:
        self.family[parentName].append(childName)

    def death(self, name: str) -> None:
        self.dead.add(name)

    def getInheritanceOrder(self) -> List[str]:
        order = []
        def dfs(name: str):
          if name not in self.dead: order.append(name)
          for child in self.family[name]: dfs(child)
        dfs(self.kingName)
        return order


# Your ThroneInheritance object will be instantiated and called as such:
# obj = ThroneInheritance(kingName)
# obj.birth(parentName,childName)
# obj.death(name)
# param_3 = obj.getInheritanceOrder()

# 1601. Maximum Number of Achievable Transfer Requests
'''Solution: Combination
Try all combinations: O(2^n * (r + n))
Space complexity: O(n)'''
class Solution:
    def maximumRequests(self, n: int, requests: List[List[int]]) -> int:
        r = len(requests)
        ans = 0
        for s in range(1 << r):
          degrees = [0] * n
          for i in range(r):
            if s & (1 << i):
              degrees[requests[i][0]] -= 1
              degrees[requests[i][1]] += 1
          if not any(degrees):
            ans = max(ans, bin(s).count('1'))
        return ans

'''Check All Combinations

Intuition
We can brute forces all combinations of requests,
and then check if it's achievable.


Explanation
For each combination, use a mask to present the picks.
The kth bits means we need to satisfy the kth request.

If for all buildings, in degrees == out degrees,
it's achievable.


Complexity
Time O((N + R) * 2^R)
Space O(N)


Solution 1'''
class Solution(object):
    def maximumRequests(self, n, requests):
        """
        :type n: int
        :type requests: List[List[int]]
        :rtype: int
        """
        nr = len(requests)
        res = 0

        def test(mask):
            outd = [0] * n
            ind = [0] * n
            for k in xrange(nr):
                if (1 << k) & mask:
                    outd[requests[k][0]] += 1
                    ind[requests[k][1]] += 1
            return sum(outd) if outd == ind else 0

        for i in xrange(1 << nr):
            res = max(res, test(i))
        return res
# Solution 2: Using Combination
class Solution(object):
    def maximumRequests(self, n, requests):
        """
        :type n: int
        :type requests: List[List[int]]
        :rtype: int
        """
        for k in range(len(requests), 0, -1):
            for c in itertools.combinations(range(len(requests)), k):
                degree = [0] * n
                for i in c:
                    degree[requests[i][0]] -= 1
                    degree[requests[i][1]] += 1
                if not any(degree):
                    return k
        return 0
# Solution 3: Using Counter
class Solution:
    def maximumRequests(self, n: int, requests: List[List[int]]) -> int:
        for k in range(len(requests), 0, -1):
            for c in combinations(requests, k):
                if Counter(a for a, b in c) == Counter(b for a, b in c):
                    return k
        return 0     
# 1604. Alert Using Same Key-Card Three or More Times in a One Hour Period

# 1605. Find Valid Matrix Given Row and Column Sums

# 1593. Split a String Into the Max Number of Unique Substrings

# 1594. Maximum Non Negative Product in a Matrix

# 1595. Minimum Cost to Connect Two Groups of Points

# 1598. Crawler Log Folder

# 1599. Maximum Profit of Operating a Centennial Wheel

# 1589. Maximum Sum Obtained of Any Permutation

# 1590. Make Sum Divisible by P
'''Solution: HashTable + Prefix Sum
Very similar to subarray target sum.

Basically, we are trying to find a shortest subarray 
that has sum % p equals to r = sum(arr) % p.

We use a hashtable to store the last index of the prefix sum % p 
and check whether (prefix_sum + p – r) % p exists or not.

Time complexity: O(n)
Space complexity: O(n)'''
class Solution:
    def minSubarray(self, nums: List[int], p: int) -> int:
        r = sum(nums) % p
        if r == 0: return 0
        m = {0: -1}
        ans = len(nums)
        s = 0
        for i, x in enumerate(nums):
          s = (s + x) % p
          t = (s + p - r) % p
          if t in m:
            ans = min(ans, i - m[t])
          m[s] = i
        return -1 if ans == len(nums) else ans

# 1591. Strange Printer II
''' Straight Forward

Explanation
For each color, find its edge most index.
Then we need to paint this color from [top, left] to [bottom, right].

If in the rectangle, all the colors are either the same or 0,
we mark all of them to 0.

If we can mark the whole grid to 0, it means the target if printable.


Complexity
Time O(CCMN)
Space O(4N)'''
class Solution(object):
    def isPrintable(self, A):
        """
        :type targetGrid: List[List[int]]
        :rtype: bool
        """
        m, n = len(A), len(A[0])
        pos = [[m, n, 0, 0] for i in xrange(61)]
        colors = set()
        for i in xrange(m):
            for j in xrange(n):
                c = A[i][j]
                colors.add(c)
                pos[c][0] = min(pos[c][0], i)
                pos[c][1] = min(pos[c][1], j)
                pos[c][2] = max(pos[c][2], i)
                pos[c][3] = max(pos[c][3], j)

        def test(c):
            for i in xrange(pos[c][0], pos[c][2] + 1):
                for j in xrange(pos[c][1], pos[c][3] + 1):
                    if A[i][j] > 0 and A[i][j] != c:
                        return False
            for i in xrange(pos[c][0], pos[c][2] + 1):
                for j in xrange(pos[c][1], pos[c][3] + 1):
                    A[i][j] = 0
            return True

        while colors:
            colors2 = set()
            for c in colors:
                if not test(c):
                    colors2.add(c)
            if len(colors2) == len(colors):
                return False
            colors = colors2
        return True

# 1592. Rearrange Spaces Between Words

# 1579. Remove Max Number of Edges to Keep Graph Fully Traversable
'''Solution: Greedy + Spanning Tree / Union Find
Use type 3 (both) edges first.

Time complexity: O(E)
Space complexity: O(n)'''
class DSU:
  def __init__(self, n: int):
    self.p = list(range(n))
    self.e = 0
    
  def find(self, x: int) -> int:
    if x != self.p[x]: self.p[x] = self.find(self.p[x])
    return self.p[x]
  
  def merge(self, x: int, y: int) -> int:
    rx, ry = self.find(x), self.find(y)
    if rx == ry: return 1
    self.p[rx] = ry
    self.e += 1
    return 0
  
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        A, B = DSU(n + 1), DSU(n + 1)    
        ans = 0
        for t, x, y in edges:
          if t != 3: continue
          ans += A.merge(x, y)
          B.merge(x, y)
        for t, x, y in edges:
          if t == 3: continue
          d = A if t == 1 else B
          ans += d.merge(x, y)
        return ans if A.e == B.e == n - 1 else -1

# 1582. Special Positions in a Binary Matrix

# 1583. Count Unhappy Friends

# 1584. Min Cost to Connect All Points

# 1585. Check If String Is Transformable With Substring Sort Operations
'''Solution: Queue

We can move a smaller digit from right to left by sorting two adjacent digits.
e.g. 18572 -> 18527 -> 18257 -> 12857, 
but we can not move a larger to the left of a smaller one.

Thus, for each digit in the target string, 
we find the first occurrence of it in s, 
and try to move it to the front by checking if there is any smaller one in front of it.

Time complexity: O(n)
Space complexity: O(n)
'''
class Solution:
    def isTransformable(self, s: str, t: str) -> bool:
        idx = defaultdict(deque)
        for i, c in enumerate(s):
          idx[int(c)].append(i)
        for c in t:
          d = int(c)
          if not idx[d]: return False
          for i in range(d):
            if idx[i] and idx[i][0] < idx[d][0]: return False
          idx[d].popleft()
        return True

# 1575. Count All Possible Routes
'''Solution: DP
dp[j][f] := # of ways to start from city ‘start’ to reach city ‘j’ with fuel level f.

dp[j][f] = sum(dp[i][f + d]) d = dist(i, j)

init: dp[start][fuel] = 1

Time complexity: O(n^2*fuel)
Space complexity: O(n*fuel)

'''
class Solution:
    def countRoutes(self, locations: List[int], start: int, finish: int, fuel: int) -> int:
        @lru_cache(None)
        def dp(i, f): # ways to reach |finsh| from |i| with |f| fuel.
          if f < 0: return 0
          return (sum(dp(j, f - abs(locations[i] - locations[j])) 
                     for j in range(len(locations)) if i != j) + (i == finish)) % (10**9 + 7)
        return dp(start, fuel)
        
# 1576. Replace All ?'s to Avoid Consecutive Repeating Characters

# 1577. Number of Ways Where Square of Number Is Equal to Product of Two Numbers

# 1578. Minimum Time to Make Rope Colorful
'''Solution: Group by group
For a group of same letters, delete all expect the one with the highest cost.

Time complexity: O(n)
Space complexity: O(1)'''
class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        colors = '*' + colors + '*'
        neededTime = [0] + neededTime + [0]
        ans = t = m = 0    
        for i in range(1, len(colors)):
          if colors[i] != colors[i - 1]:
            ans += t - m
            t = m = 0
          t += neededTime[i]
          m = max(m, neededTime[i])
        return ans

# 1573. Number of Ways to Split a String
# One pass: Space complexity: O(n)
class Solution:
    def numWays(self, s: str) -> int:
        n = len(s)
        p = defaultdict(int)
        c = 0
        for ch in s:
          if ch == '1': c += 1
          p[c] += 1
        if c % 3 != 0: return 0
        if c == 0: return ((n - 1) * (n - 2) # 2) % (10**9 + 7)
        return (p[c # 3] * p[c # 3 * 2]) % (10**9 + 7)
'''Solution: Counting

Count how many ones in the binary string as T, if not a factor of 3, 
then there is no answer.

Count how many positions that have prefix sum of T/3 as l, 
and how many positions that have prefix sum of T/3*2 as r.

Ans = l * r

But we need to special handle the all zero cases, 
which equals to C(n-2, 2) = (n – 1) * (n – 2) / 2

Time complexity: O(n)
Space complexity: O(1)'''

class Solution:
    def numWays(self, s: str) -> int:
        n = len(s)
        t = s.count('1')
        if t % 3 != 0: return 0
        if t == 0: return ((n - 1) * (n - 2) # 2) % (10**9 + 7)
        t #= 3
        l, r, c = 0, 0, 0
        for i, ch in enumerate(s):
          if ch == '1': c += 1
          if c == t: l += 1
          elif c == t * 2: r += 1
        return (l * r) % (10**9 + 7) 

# 1569. Number of Ways to Reorder Array to Get Same BST
'''Solution: Recursion + Combinatorics

For a given root (first element of the array), 
we can split the array into left children (nums[i] < nums[0]) 
and right children (nums[i] > nums[0]). 
Assuming there are l nodes for the left and r nodes for the right. 
We have C(l + r, l) different ways to insert l elements into a (l + r) sized array. 
Within node l / r nodes, we have ways(left) / ways(right) different ways to 
re-arrange those nodes. 
So the total # of ways is:
C(l + r, l) * ways(l) * ways(r)
Don’t forget to minus one for the final answer.

Time complexity: O(n^2)
Space complexity: O(n^2)

'''
class Solution:
    def numOfWays(self, nums: List[int]) -> int:
        def ways(nums):
          if len(nums) <= 2: return 1
          l = [x for x in nums if x < nums[0]]
          r = [x for x in nums if x > nums[0]]
          return comb(len(l) + len(r), len(l)) * ways(l) * ways(r)
        return (ways(nums) - 1) % (10**9 + 7)

# 1560. Most Visited Sector in a Circular Track

# 1561. Maximum Number of Coins You Can Get

# 1563. Stone Game V

# 1566. Detect Pattern of Length M Repeated K or More Times

# 1553. Minimum Number of Days to Eat N Oranges
'''Solution: Greedy + DP

Eat oranges one by one to make it a multiply of 2 or 3 
such that we can eat 50% or 66.66…% of the oranges in one step.
dp(n) := min steps to finish n oranges.
base case n <= 1, dp(n) = n
transition: dp(n) = 1 + min(n%2 + dp(n/2), n % 3 + dp(n / 3))
e.g. n = 11,
we eat 11%2 = 1 in one step, left = 10 and then eat 10 / 2 = 5 in another step. 
5 left for the subproblem.
we eat 11%3 = 2 in two steps, left = 9 and then eat 9 * 2 / 3 = 6 in another step, 
3 left for the subproblem.
dp(11) = 1 + min(1 + dp(5), 2 + dp(3))

T(n) = 2*T(n/2) + O(1) = O(n)
Time complexity: O(n) # w/o memoization, close to O(logn) in practice.
Space complexity: O(logn)'''
class Solution:
    def minDays(self, n: int) -> int:
        @lru_cache(None)
        def dp(n):
          if n <= 1: return n
          return 1 + min(n % 2 + dp(n # 2), n % 3 + dp(n # 3))
        return dp(n)

# 1556. Thousand Separator

# 1558. Minimum Numbers of Function Calls to Make Target Array
'''Solution: count 1s


For 5 (101b), we can add 1s for 5 times which of cause 
isn’t the best way to generate 5, the optimal way is to [+1, *2, +1]. 
We have to add 1 for each 1 in the binary format. 
e.g. 11 (1011), we need 3x “+1” op, and 4 “*2” op. 
Fortunately, the “*2” can be shared/delayed, 
thus we just need to find the largest number.
e.g. [2,4,8,16]
[0, 0, 0, 0] -> [0, 0, 0, 1] -> [0, 0, 0, 2]
[0, 0, 0, 2] -> [0, 0, 1, 2] -> [0, 0, 2, 4]
[0, 0, 2, 4] -> [0, 1, 2, 4] -> [0, 2, 4, 8]
[0, 2, 4, 8] -> [1, 2, 4, 8] -> [2, 4, 8, 16]
ans = sum{count_1(arr_i)} + high_bit(max(arr_i))

Time complexity: O(n*log(max(arr_i))
Space complexity: O(1)'''
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        return sum(bin(x).count('1') for x in nums) + len(bin(max(nums))) - 3

# 1559. Detect Cycles in 2D Grid

# 1546. Maximum Number of Non-Overlapping Subarrays With Sum Equals Target
 
# 1547. Minimum Cost to Cut a Stick
'''Solution: Range DP
dp[i][j] := min cost to finish the i-th cuts to the j-th (in sorted order)
dp[i][j] = r – l + min(dp[i][k – 1], dp[k + 1][j]) 
# [l, r] is the current stick range.

Time complexity: O(n^3)
Space complexity: O(n^2)'''
class Solution:
    def minCost(self, n: int, cuts: List[int]) -> int:
        @lru_cache(maxsize=None)
        def dp(i, j, l, r):
          if i > j: return 0
          if i == j: return r - l
          return r - l + min(dp(i, k - 1, l, cuts[k])
                           + dp(k + 1, j, cuts[k], r) 
                             for k in range(i, j + 1))
        cuts.sort()
        return dp(0, len(cuts) - 1, 0, n)

# 1550. Three Consecutive Odds
 
# 1551. Minimum Operations to Make Array Equal

# 1545. Find Kth Bit in Nth Binary String
'''Solution 2: Recursion
All the strings have odd length of L = (1 << n) – 1,
Let say the center m = (L + 1) / 2
if n == 1, k should be 1 and ans is “0”.
Otherwise
if k == m, we know it’s “1”.
if k < m, the answer is the same as find(n-1, K)
if k > m, we are finding a flipped and mirror char in S(n-1), 
thus the answer is flip(find(n-1, L – k + 1)).

Time complexity: O(n)
Space complexity: O(n)'''
class Solution:
    def findKthBit(self, n: int, k: int) -> str:
        if n == 1: return "0"
        l = (1 << n) - 1    
        if k == (l + 1) / 2:
          return "1"
        elif k < (l + 1) / 2:
          return self.findKthBit(n - 1, k)
        else:
          return str(1 - int(self.findKthBit(n - 1, l - k + 1)))

# 1544. Make The String Great
'''Solution: Stack
Iterator over the string, compare current char with top of the stack, 
if they are a bad pair, pop the stack (remove both of them).
Otherwise, push the current char onto the stack.

input: “abBAcC”
“a”
“ab”
“abB” -> “a”
“aA” -> “”
“c”
“cC” -> “”
ans = “”

Time complexity: O(n)
Space complexity: O(n)

'''
class Solution:
    def makeGood(self, s: str) -> str:
        ans = []
        for c in s:
          if ans and abs(ord(ans[-1]) - ord(c)) == 32:
            ans.pop()
          else:
            ans.append(c)
        return "".join(ans)

# 1542. Find Longest Awesome Substring
'''Solution: Prefix mask + Hashtable


For a palindrome all digits must occurred even times expect one. 
We can use a 10 bit mask to track the occurrence of each digit for prefix s[0~i]. 
0 is even, 1 is odd.

We use a hashtable to track the first index of each prefix state.
If s[0~i] and s[0~j] have the same state 
which means every digits in s[i+1~j] occurred even times (zero is also even) 
and it’s an awesome string. 
Then (j – (i+1) + 1) = j – i is the length of the palindrome. So far so good.

But we still need to consider the case when there is a digit with odd occurrence. 
We can enumerate all possible ones from 0 to 9, 
and temporarily flip the bit of the digit and see whether that state happened before.

fisrt_index[0] = -1, first_index[*] = inf
ans = max(ans, j – first_index[mask])

Time complexity: O(n)
Space complexity: O(2^10) = O(1)'''
class Solution:
    def longestAwesome(self, s: str) -> int:
        idx = [-1] + [len(s)] * 1023
        ans, mask = 0, 0
        for i, c in enumerate(s):
          mask ^= 1 << (ord(c) - ord('0'))
          ans = max([ans, i - idx[mask]]
                    + [i - idx[mask ^ (1 << j)] for j in range(10)])
          idx[mask] = min(idx[mask], i)
        return ans

# 1541. Minimum Insertions to Balance a Parentheses String

# 1540. Can Convert String in K Moves

# 1503. Last Moment Before All Ants Fall Out of a Plank
'''Solution: Keep Walking
When two ants A –> and <– B meet at some point, 
they change directions <– A B –>, 
we can swap the ids of the ants as <– B A–>, 
so it’s the same as walking individually and passed by. 
Then we just need to find the max/min of the left/right arrays.

Time complexity: O(n)
Space complexity: O(1)'''
class Solution:
    def getLastMoment(self, n: int, left: List[int], right: List[int]) -> int:
        t1 = max(left) if left else 0
        t2 = n - min(right) if right else 0
        return max(t1, t2)

# 1504. Count Submatrices With All Ones

# 1505. Minimum Possible Integer After at Most K Adjacent Swaps On Digits
'''Solution 2: Binary Indexed Tree / Fenwick Tree

Moving elements in a string is a very expensive operation, 
basically O(n) per op. Actually, we don’t need to move the elements physically, 
instead we track how many elements before i has been moved to the “front”. 
Thus we know the cost to move the i-th element to the “front”, 
which is i – elements_moved_before_i or prefix_sum(0~i-1) 
if we mark moved element as 1.

We know BIT / Fenwick Tree is good for dynamic prefix sum computation 
which helps to reduce the time complexity to O(nlogn).

Time complexity: O(nlogn)
Space complexity: O(n)'''
class Fenwick:
  def __init__(self, n):
    self.sums = [0] * (n + 1)
  
  def query(self, i):
    ans = 0
    i += 1
    while i > 0:
      ans += self.sums[i];
      i -= i & -i
    return ans
  
  def update(self, i, delta):
    i += 1
    while i < len(self.sums):
      self.sums[i] += delta
      i += i & -i
 
 
class Solution:
  def minInteger(self, num: str, k: int) -> str:    
    n = len(num)
    used = [False] * n
    pos = [deque() for _ in range(10)]
    for i, c in enumerate(num):
      pos[ord(c) - ord("0")].append(i)
    tree = Fenwick(n)
    ans = []
    while k > 0 and len(ans) < n:
      for d in range(10):
        if not pos[d]: continue
        i = pos[d][0]
        cost = i - tree.query(i - 1)
        if cost > k: continue
        k -= cost
        ans.append(chr(d + ord("0")))
        tree.update(i, 1)
        used[i] = True
        pos[d].popleft()
        break
    for i in range(n):
      if not used[i]: ans.append(num[i])
    return "".join(ans)

# 1494. Parallel Courses II

# 1496. Path Crossing

# 1497. Check If Array Pairs Are Divisible by k

# 1499. Max Value of Equation
'''Solution 2: Monotonic Queue

Maintain a monotonic queue:
1. The queue is sorted by y – x in descending order.
2. Pop then front element when xj – x_front > k, they can’t be used anymore.
3. Record the max of {xj + yj + (y_front – x_front)}
4. Pop the back element when yj – xj > y_back – x_back, 
they are smaller and lefter. Won’t be useful anymore.
5. Finally, push the j-th element onto the queue.

Time complexity: O(n)
Space complexity: O(n)

'''
class Solution:
    def findMaxValueOfEquation(self, points: List[List[int]], k: int) -> int:
        ans = float('-inf')
        q = deque() # {(y - x, x)}
        for x, y in points:
          while q and x - q[0][1] > k: q.popleft()
          if q: ans = max(ans, x + y + q[0][0])
          while q and y - x >= q[-1][0]: q.pop()
          q.append((y - x, x))
        return ans

# 1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree

# 1492. The kth Factor of n

# 1493. Longest Subarray of 1's After Deleting One Element

# 1487. Making File Names Unique

# 1486. XOR Operation in an Array

# 1481. Least Number of Unique Integers after K Removals

# 1478. Allocate Mailboxes
'''DP Solution

Explantion
dp[i] will means that,
the minimum distance of i + 1 first house.

B[i] = A[0] + A[1] + A[2] + .. + A[i-1]
cal(i, j) will return the minimum distance,
between A[i]~A[j] with only one mailbox.

Initialy, when k = 1, dp[i] = cal(0, i)
when we increment k, we can update dp with one more mailbox added.

Here we brute force the number of houses that new mailbox in charge.
The brute force here are good enough to get accepted.


Explantion for the cost
What and why is last = (B[j + 1] - B[m2]) - (B[m1 + 1] - B[i + 1]);

All from @caohuicn:

First of all,
last is to calculate distances for houses [i + 1, j] with 1 mailbox,
since the dp recurrence relation is:
dp[j][k] = min(dp[i][k-1] + cal[i + 1, j])
(second dimension is implicit in the code);

For houses [i + 1, j],
if the number of houses is odd,
the only mailbox should always be placed in the middle house.
If number of house is even, it can be placed anywhere
between the middle 2 houses.

Let's say we always put it at m1;
Now let's see the meaning of m1 and m2. For even houses,
m1 + 1 == m2, for odd houses, m1 == m2.
The point of introducing 2 variables is to
make sure number of houses between [i+1, m1] and [m2,j] are always equal.
(B[j + 1] - B[m2]) means A[m2] + A[m2+1] + ... + A[j],
(B[m1 + 1] - B[i + 1]) means A[i+1] + A[i+2] + ... + A[m1],
so last becomes A[j] - A[i+1] + A[j-1] - A[i+2] +... + A[m2] - A[m1].

We can interpret it as:
if the mailbox is placed between any 2 houses x and y,
the sum of distances for both houses will be A[y] - A[x].
Say we have 2n houses, then there will be n pairs;
if we have 2n + 1 houses, then there will n + 1 pairs.
Another way to interpret it is:
if the mailbox is placed at m1,
for all the right side houses,
the sum of distances will be
A[m2]-A[m1] + A[m2+1]-A[m1] + ... + A[j]-A[m1],
and for the left side (including m1),
it'll be A[m1]-A[i+1]+A[m1]-A[i+2]+...+A[m1]-A[m1-1] + A[m1]-A[m1].

Adding these 2 things together,
A[m1]s will be cancelled out,
since number of houses between [i+1, m1] and [m2,j] are always equal.

Hope it helps.

Complexity
Time O(KNN)
Space O(N)

Note that solution O(KN) is also possible to come up with.'''
class Solution(object):
    def minDistance(self, A, k):
        """
        :type houses: List[int]
        :type k: int
        :rtype: int
        """
        
        A.sort()
        n = len(A)
        B = [0]
        for i, a in enumerate(A):
            B.append(B[i] + a)

        def cal(i, j):
            m1, m2 = (i + j) / 2, (i + j + 1) / 2
            return (B[j + 1] - B[m2]) - (B[m1 + 1] - B[i])

        dp = [cal(0, j) for j in xrange(n)]
        for k in xrange(2, k + 1):
            for j in xrange(n - 1, k - 2, -1):
                for i in xrange(k - 2, j):
                    dp[j] = min(dp[j], dp[i] + cal(i + 1, j))
        return int(dp[-1])

# 1476. Subrectangle Queries

# 1477. Find Two Non-overlapping Sub-arrays Each With Target Sum

# 1475. Final Prices With a Special Discount in a Shop

# 1537. Get the Maximum Score
'''Solution: Two Pointers + DP
Since numbers are strictly increasing, 
we can always traverse the smaller one using two pointers.
Traversing ([2,4,5,8,10], [4,6,8,10])
will be like [2, 4/4, 5, 6, 8, 10/10]
It two nodes have the same value, we have two choices and pick the larger one, 
then both move nodes one step forward. 
Otherwise, the smaller node moves one step forward.
dp1[i] := max path sum ends with nums1[i-1]
dp2[j] := max path sum ends with nums2[j-1]
if nums[i -1] == nums[j – 1]:
dp1[i] = dp2[j] = max(dp[i-1], dp[j-1]) + nums[i -1]
i += 1, j += 1
else if nums[i – 1] < nums[j – 1]:
dp[i] = dp[i-1] + nums[i -1]
i += 1
else if nums[j – 1] < nums[i – 1]:
dp[j] = dp[j-1] + nums[j -1]
j += 1
return max(dp1[-1], dp2[-1])

Time complexity: O(n)
Space complexity: O(n) -> O(1)'''
class Solution:
    def maxSum(self, nums1: List[int], nums2: List[int]) -> int:
        n1, n2 = len(nums1), len(nums2)
        i, j = 0, 0
        a, b = 0, 0
        while i < n1 or j < n2:
          if i < n1 and j < n2 and nums1[i] == nums2[j]:
            a = b = max(a, b) + nums1[i]
            i += 1
            j += 1
          elif i < n1 and (j == n2 or nums1[i] < nums2[j]):
            a += nums1[i]
            i += 1
          else:
            b += nums2[j]
            j += 1
        return max(a, b) % (10**9 + 7)  

# 1536. Minimum Swaps to Arrange a Binary Grid

# 1535. Find the Winner of an Array Game
class Solution:
    def getWinner(self, arr: List[int], k: int) -> int:
        winner = arr[0]
        win = 0
        for x in arr[1:]:
          if x > winner:
            winner, win = x, 0
          win += 1
          if win == k: break
        return winner

# 1534. Count Good Triplets

# 1528. Shuffle String
'''Solution: Simulation
Time complexity: O(n)
Space complexity: O(n)'''
class Solution:
    def restoreString(self, s: str, indices: List[int]) -> str:
        ans = [None] * len(s)
        for i, idx in enumerate(indices):
          ans[idx] = s[i]
        return ''.join(ans)

# 1530. Number of Good Leaf Nodes Pairs

# 1529. Minimum Suffix Flips

# 1531. String Compression II
'''State compression

dp[i][k] := min len of s[i:] encoded by deleting at most k charchters.

dp[i][k] = min(dp[i+1][k-1] # delete s[i]
encode_len(s[i~j] == s[i]) + dp(j+1, k – sum(s[i~j])) for j in range(i, n)) # keep

Time complexity: O(n^2*k)
Space complexity: O(n*k)'''
class Solution:
    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        n = len(s)
        @functools.lru_cache(maxsize=None)
        def dp(i, k):
          if k < 0: return n
          if i + k >= n: return 0
          ans = dp(i + 1, k - 1)
          l = 0
          same = 0
          for j in range(i, n):
            if s[j] == s[i]:
              same += 1
              if same <= 2 or same == 10 or same == 100:
                l += 1
            diff = j - i + 1 - same
            if diff < 0: break
            ans = min(ans, l + dp(j + 1, k - diff))
          return ans
        return dp(0, k)

# 1513. Number of Substrings With Only 1s
'''Solution: DP / Prefix Sum
dp[i] := # of all 1 subarrays end with s[i].
dp[i] = dp[i-1] if s[i] == ‘1‘ else 0
ans = sum(dp)
s=1101
dp[0] = 1 // 1
dp[1] = 2 // 11, *1
dp[2] = 0 // None
dp[3] = 1 // ***1
ans = 1 + 2 + 1 = 5

Time complexity: O(n)
Space complexity: O(n)

dp[i] only depends on dp[i-1], we can reduce the space complexity to O(1)'''
class Solution:
    def numSub(self, s: str) -> int:
        kMod = 10**9 + 7
        ans = 0
        cur = 0
        for c in s:
          cur = cur + 1 if c == '1' else 0
          ans += cur
        return ans % kMod

# 1521. Find a Value of a Mysterious Function Closest to Target

# 1524. Number of Sub-arrays With Odd Sum
'''Solution: DP

We would like to know how many subarrays end with arr[i] have odd or even sums.

dp[i][0] := # end with arr[i] has even sum
dp[i][1] := # end with arr[i] has even sum

if arr[i] is even:

  dp[i][0]=dp[i-1][0] + 1, dp[i][1]=dp[i-1][1]

else:

  dp[i][1]=dp[i-1][0], dp[i][0]=dp[i-1][0] + 1

ans = sum(dp[i][1])

Time complexity: O(n)
Space complexity: O(n) -> O(1)

'''
class Solution:
    def numOfSubarrays(self, arr: List[int]) -> int:
        ans, odd, even = 0, 0, 0
        for x in arr:
          if x & 1:
            odd, even = even + 1, odd
          else:
            odd, even = odd, even + 1
          ans += odd
        return ans % int(1e9 + 7)

# 1525. Number of Good Ways to Split a String
'''Solution: Sliding Window
Count the frequency of each letter and count number of unique letters 
for the entire string as right part.
Iterate over the string, add current letter to the left part, 
and remove it from the right part.
We only
increase the number of unique letters when its frequency becomes to 1
decrease the number of unique letters when its frequency becomes to 0
Time complexity: O(n)
Space complexity: O(1)

'''
class Solution:
    def numSplits(self, s: str) -> int:
        s = [ord(c) - ord('a') for c in s]
        l, r = [0] * 26, [0] * 26
        cl, cr, ans = 0, 0, 0
        for c in s:
          r[c] += 1
          if r[c] == 1: cr += 1
        for c in s:
          l[c] += 1
          r[c] -= 1
          if l[c] == 1: cl += 1
          if r[c] == 0: cr -= 1
          if cl == cr: ans += 1
        return ans

# 1526. Minimum Number of Increments on Subarrays to Form a Target Array

# 1510. Stone Game IV
'''Solution: Recursion w/ Memoization / DP
Let win(n) denotes whether the current play will win or not.
Try all possible square numbers and see whether the other player will lose or not.
win(n) = any(win(n – i*i) == False) ? True : False
base case: win(0) = False

Time complexity: O(nsqrt(n))
Space complexity: O(n)'''
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [None] * (n + 1)
        dp[0] = False
        for i in range(0, n):      
          if dp[i]: continue
          for j in range(1, n + 1):      
            if i + j * j > n: break
            dp[i + j * j] = True
        return dp[n]

# 1518. Water Bottles
'''Solution: Simulation
Time complexity: O(logb/loge)?
Space complexity: O(1)'''
class Solution:
    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        ans = numBottles
        while numBottles >= numExchange:
            numBottles, remainder = divmod(numBottles, numExchange)
            ans += numBottles
            numBottles += remainder
        return ans

class Solution:
    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        return numBottles + (numBottles - 1) // (numExchange - 1)

# 1519. Number of Nodes in the Sub-Tree With the Same Label
'''Solution: Post order traversal + hashtable
For each label, record the count. 
When visiting a node, we first record the current count of its label as before, 
and traverse its children, when done, increment the current count, 
ans[i] = current – before.

Time complexity: O(n)
Space complexity: O(n)'''
class Solution:
    def countSubTrees(self, n: int, edges: List[List[int]], labels: str) -> List[int]:
        g = [[] for _ in range(n)]
        for u, v in edges:
          g[u].append(v)
          g[v].append(u)
        seen = [False] * n
        count = [0] * 26
        ans = [0] * n
        def postOrder(i):
          if seen[i]: return
          seen[i] = True
          before = count[ord(labels[i]) - ord('a')]
          for j in g[i]: postOrder(j)
          count[ord(labels[i]) - ord('a')] += 1
          ans[i] = count[ord(labels[i]) - ord('a')] - before
        postOrder(0)
        return ans

# 1520. Maximum Number of Non-Overlapping Substrings

# 1514. Path with Maximum Probability
'''Solution: Dijkstra’s Algorithm
max(P1*P2*…*Pn) => max(log(P1*P2…*Pn)) 
=> max(log(P1) + log(P2) + … + log(Pn) => min(-(log(P1) + log(P2) … + log(Pn)).

Thus we can convert this problem to the classic single source shortest path problem 
that can be solved with Dijkstra’s algorithm.

Time complexity: O(ElogV)
Space complexity: O(E+V)'''
class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        g = [[] for _ in range(n)]
        for i, e in enumerate(edges):
          g[e[0]].append((e[1], -math.log(succProb[i])))
          g[e[1]].append((e[0], -math.log(succProb[i])))
        seen = [False] * n
        dist = [float('inf')] * n
        dist[start] = 0.0
        q = [(dist[start], start)]
        while q:
          _, u = heapq.heappop(q)
          if seen[u]: continue
          seen[u] = True
          if u == end: return math.exp(-dist[u])
          for v, w in g[u]:
            if seen[v] or dist[u] + w > dist[v]: continue
            dist[v] = dist[u] + w        
            heapq.heappush(q, (dist[v], v))
        return 0

# 1512. Number of Good Pairs
'''Solution 2: Hashtable
Store the frequency of each number so far, when we have a number x at pos j, 
and it appears k times before. Then we can form additional k pairs.

Time complexity: O(n)
Space complexity: O(range)'''
class Solution:
    def numIdenticalPairs(self, nums: List[int]) -> int:
        f = defaultdict(int)
        ans = 0
        for x in nums:
          ans += f[x]
          f[x] += 1
        return ans

# 1507. Reformat Date
'''Solution: String + HashTable
Time complexity: O(1)
Space complexity: O(1)

'''
class Solution:
    def reformatDate(self, date: str) -> str:
        m = {"Jan": "01", "Feb": "02", "Mar": "03", 
         "Apr": "04", "May": "05", "Jun": "06", 
         "Jul": "07", "Aug": "08", "Sep": "09", 
         "Oct": "10", "Nov": "11", "Dec": "12"}
        items = date.split(" ")
        day = items[0][:-2]
        if len(day) == 1: day = "0" + day
        return items[2] + "-" + m[items[1]] + "-" + day

# 662. Maximum Width of Binary Tree
'''Solution: DFS

Let us assign an id to each node, similar to the index of a heap. 
root is 1, left child = parent * 2, right child = parent * 2 + 1. 
Width = id(right most child) – id(left most child) + 1, so far so good.
However, this kind of id system grows exponentially, 
it overflows even with long type with just 64 levels. 
To avoid that, we can remap the id with id – id(left most child of each level).

Time complexity: O(n)
Space complexity: O(h)

'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        ids = []
        def dfs(node: TreeNode, d: int, id: int) -> int:
          if not node: return 0
          if d == len(ids): ids.append(id)
          return max(id - ids[d] + 1, 
                     dfs(node.left, d + 1, (id - ids[d]) * 2),
                     dfs(node.right, d + 1, (id - ids[d]) * 2 + 1))
        return dfs(root, 0, 0)

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

# 

# 

# 

# 

#

# 

# 

# 

# 

# 

# 

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#



