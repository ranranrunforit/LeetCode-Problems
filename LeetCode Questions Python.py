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