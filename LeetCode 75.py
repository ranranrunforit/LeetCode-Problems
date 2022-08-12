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


# Day 1 Implementation/Simulation
# 202. Happy Number

# 54. Spiral Matrix

# 1706. Where Will the Ball Fall
'''
Explanation
We drop the ball at grid[0][i]
and track ball position i1, which initlized as i.

An observation is that,
if the ball wants to move from i1 to i2,
we must have the board direction grid[j][i1] == grid[j][i2]


Complexity
Time O(mn)
Space O(n)
'''
class Solution(object):
    def findBall(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: List[int]
        """
        m, n = len(grid), len(grid[0])

        def test(i):
            for j in xrange(m):
                i2 = i + grid[j][i]
                if i2 < 0 or i2 >= n or grid[j][i2] != grid[j][i]:
                    return -1
                i = i2
            return i
        return map(test, range(n))

# Day 2 String
# 14. Longest Common Prefix
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ""
        shortest = min(strs,key=len)
        for i, ch in enumerate(shortest):
            for other in strs:
                if other[i] != ch:
                    return shortest[:i]
        return shortest 

# 43. Multiply Strings

# Day 3 Linked List
# 19. Remove Nth Node From End of List

# 234. Palindrome Linked List
'''
For linked list 1->2->3->2-1, 
the code below first makes the list to be 1->2->3->2<-1 and 
the second 2->None, then make 3->None, 
for even number linked list: 1->2->2->1, make first 1->2->2<-1 and 
then the second 2->None, 
and lastly do not forget to make the first 2->None 
(If forget it still works while the idea behind is a little bit different).
'''
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        fast = slow = head
        # find the mid node
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        # reverse the second half
        node = None
        while slow:
            nxt = slow.next
            slow.next = node
            node = slow
            slow = nxt
        # compare the first and second half nodes
        while node: # while node and head:
            if node.val != head.val:
                return False
            node = node.next
            head = head.next
        return True
'''
O(n) extra space solution by using deque:'''
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        queue = collections.deque([])
        cur = head
        while cur:
            queue.append(cur)
            cur = cur.next
        while len(queue) >= 2:
            if queue.popleft().val != queue.pop().val:
                return False
        return True
'''The naive approach here would be to run through the linked list and 
create an array of its values, then compare the array to its reverse to find out if it's a palindrome. 
Though this is easy enough to accomplish, we're challenged to find an approach with a space complexity 
of only O(1) while maintaining a time complexity of O(N).

The only way to check for a palindrome in O(1) space would 
require us to be able to access both nodes for comparison at the same time, 
rather than storing values for later comparison. 
This would seem to be a challenge, as the linked list only promotes travel in one direction.

But what if it didn't?

The answer is to reverse the back half of the linked list to have 
the next attribute point to the previous node instead of the next node. 
(Note: we could instead add a prev attribute as we iterate through 
the linked list, rather than overwriting next on the back half, 
but that would technically use O(N) extra space, 
just as if we'd created an external array of node values.)

The first challenge then becomes finding the middle of the linked list in 
order to start our reversing process there. 
For that, we can look to Floyd's Cycle Detection Algorithm.

With Floyd's, we'll travel through the linked list with two pointers, 
one of which is moving twice as fast as the other. 
When the fast pointer reaches the end of the list, 
the slow pointer must then be in the middle.

With slow now at the middle, we can reverse the back half of the list with 
the help of another variable to contain a reference to the previous node 
(prev) and a three-way swap. 
Before we do this, however, we'll want to set prev.next = null, 
so that we break the reverse cycle and avoid an endless loop.

Once the back half is properly reversed and slow is once again at 
the end of the list, we can now start fast back over again at the head and 
compare the two halves simultaneously, with no extra space required.

If the two pointers ever disagree in value, we can return false, 
otherwise we can return true if both pointers reach the middle successfully.

(Note: This process works regardless of whether the length of the linked list 
is odd or even, as the comparison will stop when slow reaches the "dead-end" 
node.)'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow, fast, prev = head, head, None
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        prev, slow, prev.next = slow, slow.next, None
        while slow:
            slow.next, prev, slow = prev, slow, slow.next
        fast, slow = head, prev
        while slow:
            if fast.val != slow.val: return False
            fast, slow = fast.next, slow.next
        return True
# Day 4 Linked List
# 328. Odd Even Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        
        p1, p2 = head, head.next
        pre = p2
        while p2 != None and p2.next != None:
            p1.next = p2.next
            p1 = p1.next
            p2.next = p1.next
            p2 = p2.next
        
        p1.next = pre
        return head
# 148. Sort List
'''
Solution: Merge Sort
Top-down (recursion)

Time complexity: O(nlogn)
Space complexity: O(logn)'''
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        def merge(l1, l2):
            dummy = ListNode(0)
            tail = dummy
            while l1 and l2:
                if l1.val > l2.val: l1, l2 = l2, l1
                tail.next = l1
                l1 = l1.next
                tail = tail.next
            tail.next = l1 if l1 else l2
            return dummy.next

        if not head or not head.next: return head
        slow = head
        fast = head.next
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        mid = slow.next
        slow.next = None
        return merge(self.sortList(head), self.sortList(mid))

# Day 5 Greedy
# 2131. Longest Palindrome by Concatenating Two Letter Words
'''
Explanation:

2 letter words can be of 2 types:

Where both letters are same
Where both letters are different
Based on the above information:

If we are able to find the mirror of a word, ans += 4
The variable unpaired is used to store the number of unpaired words 
with both letters same.
Unpaired here means a word that has not found its mirror word.
At the end if unpaired same letter words are > 0, 
we can use one of them as the center of the palindromic string.
'''
class Solution(object):
    def longestPalindrome(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        counter, ans = [[0] * 26 for _ in range(26)], 0
        for w in words:
            a, b = ord(w[0]) - ord('a'), ord(w[1]) - ord('a')
            if counter[b][a]:
                ans += 4
                counter[b][a] -= 1
            else: counter[a][b] += 1
        for i in range(26):
            if counter[i][i]:
                ans += 2
                break
        return ans

# 621. Task Scheduler
'''
Sample run for understanding how heap and temp are involved in calculating the optimal number of intervals.

Legend: '->' : heap pop operation
'#' : idle time

    [A A A A B B D E]
    using most frequent first: [A ## A ## A ## A]
    other jobs in idle time:   [A BD A BE A ## A] i.e. 10 intervals
    
    HEAP            TEMP                    TIME
    (-4,A)->        [(-3,A)]                      1
    (-2,B)->        [(-3,A), (-1,B)]          2
    (-1,D)->        [(-3,A), (-1,B)]          3
    
    *********************COOL TIME COMPLETE************************
    put items in TEMP back to HEAP
    
    HEAP becomes: [(-3,A), (-1,E), (-1,B)]
    
    HEAP            TEMP                    TIME
    (-3,A)->        [(-2,A)]                  4
    (-1,B)->        [(-2,A)]                  5
    (-1,E)->        [(-2,A)]                  6
    
    *********************COOL TIME COMPLETE************************
    put items in TEMP back to HEAP
    
    HEAP becomes: [(-2,A)]
    
    HEAP            TEMP                    TIME
    (-2,A)->        [(-1,A)]                  7
    EMPTY                                         8
    EMPTY                                         9
    
    *********************COOL TIME COMPLETE************************
    put items in TEMP back to HEAP
    
    HEAP becomes: [(-1,A)]
    
    HEAP            TEMP                    TIME
    (-1,A)->                                       10
    EMPTY         EMPTY                  break
'''
class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        curr_time, h = 0, []
        for k,v in Counter(tasks).items():
            heappush(h, (-1*v, k))
        while h:
            i, temp = 0, []
            while i <= n:
                curr_time += 1
                if h:
                    x,y = heappop(h)
                    if x != -1:
                        temp.append((x+1,y))
                if not h and not temp:
                    break
                else:
                    i += 1
            for item in temp:
                heappush(h, item)
        return curr_time
# Day 6 Tree
# 226. Invert Binary Tree

# 110. Balanced Binary Tree

# Day 7 Tree
# 543. Diameter of Binary Tree
'''
Solution 3:

Simulate recursion with a stack. 
We also need to track the return value of each node.
'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        d = {None: -1}
        s = [root]
        ans = 0
        while s:
            node = s[-1]
            if node.left in d and node.right in d:
                s.pop()
                l = d[node.left] + 1
                r = d[node.right] + 1
                ans = max(ans, l + r)
                d[node] = max(l, r)
            else:
                if node.left: s.append(node.left)
                if node.right: s.append(node.right)
        return ans

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.best = 1
        def depth(root):
            if not root: return 0
            ansL = depth(root.left)
            ansR = depth(root.right)
            self.best = max(self.best, ansL + ansR + 1)
            return 1 + max(ansL, ansR)

        depth(root)
        return self.best - 1

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.ans = 0
        
        def depth(p):
            if not p: return 0
            left, right = depth(p.left), depth(p.right)
            self.ans = max(self.ans, left+right)
            return 1 + max(left, right)
            
        depth(root)
        return self.ans

# 437. Path Sum III
'''
2. Memorization of path sum: O(n)
2.1 High level walk through
In order to optimize from the brutal force solution, we will have to think of a clear way to memorize the intermediate result. Namely in the brutal force solution, we did a lot repeated calculation. For example 1->3->5, we calculated: 1, 1+3, 1+3+5, 3, 3+5, 5.
This is a classical 'space and time tradeoff': we can create a dictionary (named cache) which saves all the path sum (from root to current node) and their frequency.
Again, we traverse through the tree, at each node, we can get the currPathSum (from root to current node). If within this path, there is a valid solution, then there must be a oldPathSum such that currPathSum - oldPathSum = target.
We just need to add the frequency of the oldPathSum to the result.
During the DFS break down, we need to -1 in cache[currPathSum], because this path is not available in later traverse.
Check the graph below for easy visualization.
image
2.2 Complexity analysis:
2.2.1 Space complexity
O(n) extra space

2.2.1 Time complexity
O(n) as we just traverse once

'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def pathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: int
        """
        # define global result and path
        self.result = 0
        cache = {0:1}
        
        # recursive to get result
        self.dfs(root, targetSum, 0, cache)
        
        # return result
        return self.result
    
    def dfs(self, root, target, currPathSum, cache):
        # exit condition
        if root is None:
            return  
        # calculate currPathSum and required oldPathSum
        currPathSum += root.val
        oldPathSum = currPathSum - target
        # update result and cache
        self.result += cache.get(oldPathSum, 0)
        cache[currPathSum] = cache.get(currPathSum, 0) + 1
        
        # dfs breakdown
        self.dfs(root.left, target, currPathSum, cache)
        self.dfs(root.right, target, currPathSum, cache)
        # when move to a different branch, the currPathSum is no longer available, hence remove one. 
        cache[currPathSum] -= 1

# Day 8 Binary Search
# 74. Search a 2D Matrix

# 33. Search in Rotated Sorted Array

# Day 9 Binary Search Tree
# 108. Convert Sorted Array to Binary Search Tree

# 230. Kth Smallest Element in a BST

# 173. Binary Search Tree Iterator

# Day 10 Graph/BFS/DFS
# 994. Rotting Oranges

# 417. Pacific Atlantic Water Flow

# Day 11 Graph/BFS/DFS
# 210. Course Schedule II
class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        indegree = [set() for _ in range(numCourses)]
        outdegree = [[] for _ in range(numCourses)]
        for p in prerequisites:
            indegree[p[0]].add(p[1])
            outdegree[p[1]].append(p[0])
        ret, start = [], [i for i in range(numCourses) if not indegree[i]]
        while start: # start contains courses without prerequisites
            newStart = [] 
            for i in start:
                ret.append(i)
                for j in outdegree[i]:
                    indegree[j].remove(i)
                    if not indegree[j]:
                        newStart.append(j)
            start = newStart # newStart contains new courses with no prerequisites
        return ret if len(ret) == numCourses else [] # can finish if ret contains all courses 
# 815. Bus Routes
'''BFS Solution
Explanation:
The first part loop on routes and record stop to routes mapping in to_route.
The second part is general bfs. Take a stop from queue and find all connected route.
The hashset seen record all visited stops and we won't check a stop for twice.
We can also use a hashset to record all visited routes, or just clear a route after visit.
'''
class Solution(object):
    def numBusesToDestination(self, routes, source, target):
        """
        :type routes: List[List[int]]
        :type source: int
        :type target: int
        :rtype: int
        """
        to_routes = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for j in route:
                to_routes[j].add(i)
        bfs = [(source, 0)]
        seen = set([source])
        for stop, bus in bfs:
            if stop == target: return bus
            for i in to_routes[stop]:
                for j in routes[i]:
                    if j not in seen:
                        bfs.append((j, bus + 1))
                        seen.add(j)
                routes[i] = []  # seen route
        return -1

# Day 12 Dynamic Programming
# 198. House Robber

# 322. Coin Change

# Day 13 Dynamic Programming
# 416. Partition Equal Subset Sum
'''
 Solution - V (Dynamic Programming using bitmask)

We can use bitmasking to condense the inner loop of previous approach into a single bit-shift operation. Here we will use bitset in C++ consisting of sum number of bits (other language can use bigInt or whatever support is provided for such operations).

Each bit in bitset (dp[i]) will denote whether sum i is possible or not. Now, when we get a new number num, it can be added to every sum already possible, i.e, every dp[i] bit which is already 1. This operation can be performed using bit shift as dp << num. How? See the following example

Suppose current dp = 1011
This means sums 0, 1 and 3 are possible to achieve.
Let the next number we get: num = 5. 
Now we can achieve (0, 1 & 3) which were already possible and (5, 6, 8) which are new sum after adding 'num=5' to previous sums

1. 'dp << num': This operation will add num to every bit .
                         3 2 1 0                                8 7 6 5 4 3 2 1 0                     
                So, dp = 1 0 1 1 will be transformed to  dp  =  1 0 1 1 0 0 0 0 0   (after 5 shifts to left)
			    Note that new dp now denotes 5, 6, 8 which are the new sums possible.
			    We will combine it with previous sums using '|' operation
				
                      8 7 6 5 4 3 2 1 0
2. 'dp | dp << num' = 1 0 1 1 0 1 0 1 1

And now we have every possible sum after combining new num with previous possible sums.
Finally, we will return dp[halfSum] denoting whether half sum is achievable or not.
Time Complexity : O(N*sum)
Space Complexity : O(sum)
'''
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        total_sum = sum(nums)
        if total_sum & 1: return False
        half_sum, dp = total_sum // 2, 1
        for num in nums:
            dp |= dp << num
        return dp & 1 << half_sum
'''
Solution - IV (Dynamic Programming - Tabulation)

We can convert the dp approach to iterative version. Here we will again use dp 
array, where dp[sum] will denote whether sum is achievable or not. 
Initially, we have dp[0] = true since a 0 sum is always achievable. 
Then for each element num, we will iterate & find if it is possible to form a 
sum j by adding num to some previously formable sum.

One thing to note that it is essential to iterate from right to left 
in the below inner loop to avoid marking multiple sum, 
say j1 as achievable and then again using that result to mark another bigger sum j2 (j2=j1+num) 
as achievable. 
This would be wrong since it would mean choosing num multiple times. 
So we start from right to left to avoid overwriting previous results updated 
in the current loop.
Time Complexity : O(N*sum)
Space Complexity : O(sum)
'''
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        total_sum = sum(nums)
        if total_sum & 1: return False
        half_sum = total_sum // 2
        dp = [True] + [False]*half_sum
        for num in nums:
            for j in range(half_sum, num-1, -1):
                dp[j] |= dp[j-num]
        return dp[half_sum]

# 152. Maximum Product Subarray

# Day 14 Sliding Window/Two Pointer
# 3. Longest Substring Without Repeating Characters

# 16. 3Sum Closest
'''Solution: Sorting + Two Pointers
Similar to LeetCode 15. 3Sum

Time complexity: O(n^2)
Space complexity: O(1)'''
class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        n = len(nums)
        d = 2**32
        ans = 0
        for i in range(n - 2):
            s, t = i + 1, n - 1
            while s < t:
                sum3 = nums[i] + nums[s] + nums[t]
                if sum3 == target: return target
                diff = abs(sum3 - target)
                if diff < d: ans, d = sum3, diff        
                if sum3 > target: 
                    t -= 1 
                else: 
                    s += 1
        return ans

# 76. Minimum Window Substring
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        target_letter_counts = collections.Counter(t)
        start = 0
        end = 0
        min_window = ""
        target_len = len(t)        
        
        for end in range(len(s)):
			# If we see a target letter, decrease the total target letter count
            if target_letter_counts[s[end]] > 0:
                target_len -= 1

            # Decrease the letter count for the current letter
			# If the letter is not a target letter, the count just becomes -ve
            target_letter_counts[s[end]] -= 1
            
			# If all letters in the target are found:
            while target_len == 0:
                window_len = end - start + 1
                if not min_window or window_len < len(min_window):
					# Note the new minimum window
                    min_window = s[start : end + 1]
                    
				# Increase the letter count of the current letter
                target_letter_counts[s[start]] += 1
                
				# If all target letters have been seen and now, a target letter is seen with count > 0
				# Increase the target length to be found. This will break out of the loop
                if target_letter_counts[s[start]] > 0:
                    target_len += 1
                    
                start+=1
                
        return min_window
# Approach 2: Optimized Sliding Window
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        if not t or not s:
            return ""

        dict_t = Counter(t)

        required = len(dict_t)

        # Filter all the characters from s into a new list along with their index.
        # The filtering criteria is that the character should be present in t.
        filtered_s = []
        for i, char in enumerate(s):
            if char in dict_t:
                filtered_s.append((i, char))

        l, r = 0, 0
        formed = 0
        window_counts = {}

        ans = float("inf"), None, None

        # Look for the characters only in the filtered list instead of entire s. This helps to reduce our search.
        # Hence, we follow the sliding window approach on as small list.
        while r < len(filtered_s):
            character = filtered_s[r][1]
            window_counts[character] = window_counts.get(character, 0) + 1

            if window_counts[character] == dict_t[character]:
                formed += 1

            # If the current window has all the characters in desired frequencies i.e. t is present in the window
            while l <= r and formed == required:
                character = filtered_s[l][1]

                # Save the smallest window until now.
                end = filtered_s[r][0]
                start = filtered_s[l][0]
                if end - start + 1 < ans[0]:
                    ans = (end - start + 1, start, end)

                window_counts[character] -= 1
                if window_counts[character] < dict_t[character]:
                    formed -= 1
                l += 1    

            r += 1    
        return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]
# Day 15 Tree
# 100. Same Tree
'''Solution: Recursion
Time complexity: O(n)
Space complexity: O(n)'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if not p and not q: return True
        if not p or not q: return False
        return all((p.val == q.val, 
                   self.isSameTree(p.left, q.left), 
                   self.isSameTree(p.right, q.right)))

class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        # p and q are both None
        if not p and not q:
            return True
        # one of p and q is None
        if not q or not p:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.right, q.right) and \
               self.isSameTree(p.left, q.left)

# 101. Symmetric Tree

# 199. Binary Tree Right Side View

# Day 16 Design
# 232. Implement Queue using Stacks

# 155. Min Stack

# 208. Implement Trie (Prefix Tree)
class Trie(object):
            
    def __init__(self):
        self.root = {}
        
    def insert(self, word):
        """
        :type word: str
        :rtype: None
        """
        p = self.root
        for c in word:            
            if c not in p: 
                p[c] = {}
            p = p[c]
        p['#'] = True

    def search(self, word):
        """
        :type word: str
        :rtype: bool
        """
        node = self.find(word)
        return node is not None and '#' in node
    
    def startsWith(self, prefix):
        return self.find(prefix) is not None
        

    def find(self, prefix):
        """
        :type prefix: str
        :rtype: bool
        """
        p = self.root
        for c in prefix:            
            if c not in p: return None
            p = p[c]
        return p

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

class Trie(object):
    class TrieNode(object):
        def __init__(self):
            self.is_word = False
            self.children = [None] * 26
            
    def __init__(self):
        self.root = Trie.TrieNode()

    def insert(self, word):
        """
        :type word: str
        :rtype: None
        """
        p = self.root
        for c in word:
            index = ord(c) - ord('a')
            if not p.children[index]: 
                p.children[index] = Trie.TrieNode()
            p = p.children[index]
        p.is_word = True

    def search(self, word):
        """
        :type word: str
        :rtype: bool
        """
        node = self.find(word)
        return node is not None and node.is_word
    
    def startsWith(self, prefix):
        return self.find(prefix) is not None
        

    def find(self, prefix):
        """
        :type prefix: str
        :rtype: bool
        """
        p = self.root
        for c in prefix:
            index = ord(c) - ord('a')
            if not p.children[index]: return None
            p = p.children[index]
        return p


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

# Day 17 Interval
# 57. Insert Interval
# O(nlgn) time, the same as Merge Intervals 
# https://leetcode.com/problems/merge-intervals/
class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        intervals.append(newInterval)
        res = []
        for i in sorted(intervals, key=lambda x:x[0]):
            if res and res[-1][-1] >= i[0]:
                res[-1][-1] = max(res[-1][-1], i[-1])
            else:
                res.append(i)
        return res

class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        index = len(intervals)
        for i in range(len(intervals)):
            if newInterval[0] < intervals[i][0]:
                index = i
                break
        
        intervals.insert(index, newInterval)
        
        ans = []
        for interval in intervals:
            if not ans or interval[0] > ans[-1][-1]:
                ans.append(interval)
            else:
                ans[-1][-1] = max(ans[-1][-1], interval[-1])
        return ans

# O(n) time, not in-place, make use of the 
# property that the intervals were initially sorted 
# according to their start times
class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        res, n = [], newInterval
        for index, i in enumerate(intervals):
            if i[-1] < n[0]:
                res.append(i)
            elif n[-1] < i[0]:
                res.append(n)
                return res+intervals[index:]  # can return earlier
            else:  # overlap case
                n[0] = min(n[0], i[0])
                n[-1] = max(n[-1], i[-1])
        res.append(n)
        return res
# 56. Merge Intervals

# Day 18 Stack
# 735. Asteroid Collision
class Solution(object):
    def asteroidCollision(self, asteroids):
        """
        :type asteroids: List[int]
        :rtype: List[int]
        """
        ans = []
        for new in asteroids:
            while ans and new < 0 < ans[-1]:
                if ans[-1] < -new:
                    ans.pop()
                    continue
                elif ans[-1] == -new:
                    ans.pop()
                break
            else:
                ans.append(new)
        return ans
# 227. Basic Calculator II
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        nums = []
        op = '+'
        cur = 0
        i = 0
        while i < len(s):
            if s[i] == ' ': 
                i += 1
                continue
            while i < len(s) and s[i].isdigit():
                cur = cur * 10 + ord(s[i]) - ord('0')
                i += 1      
            if op in '+-':
                nums.append(cur * (1 if op == '+' else -1))
            elif op == '*':
                nums[-1] *= cur
            elif op == '/':
                sign = -1 if nums[-1] < 0 or cur < 0 else 1
                nums[-1] = abs(nums[-1]) // abs(cur) * sign
            cur = 0
            if (i < len(s)): op = s[i]
            i += 1    
        return sum(nums)
# stack
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return "0"
        stack, num, sign = [], 0, "+"
        for i in xrange(len(s)):
            if s[i].isdigit():
                num = num*10+ord(s[i])-ord("0")
            if (not s[i].isdigit() and not s[i].isspace()) or i == len(s)-1:
                if sign == "-":
                    stack.append(-num)
                elif sign == "+":
                    stack.append(num)
                elif sign == "*":
                    stack.append(stack.pop()*num)
                else:
                    tmp = stack.pop()
                    if tmp//num < 0 and tmp%num != 0:
                        stack.append(tmp//num+1)
                    else:
                        stack.append(tmp//num)
                sign = s[i]
                num = 0
        return sum(stack)

# Day 19 Union Find
# 547. Number of Provinces

# 947. Most Stones Removed with Same Row or Column
'''Count the Number of Islands, O(N)
I said it's a hard problem, LC rated it as medium.

Problem:
we can remove a stone if and only if,
there is another stone in the same column OR row.
We try to remove as many as stones as possible.


One sentence to solve:
Connected stones can be reduced to 1 stone,
the maximum stones can be removed = stones number - islands number.
so just count the number of "islands".


1. Connected stones
Two stones are connected if they are in the same row or same col.
Connected stones will build a connected graph.
It's obvious that in one connected graph,
we can't remove all stones.

We have to have one stone left.
An intuition is that, in the best strategy, we can remove until 1 stone.

I guess you may reach this step when solving the problem.
But the important question is, how?


2. A failed strategy
Try to remove the least degree stone
Like a tree, we try to remove leaves first.
Some new leaf generated.
We continue this process until the root node left.

However, there can be no leaf.
When you try to remove the least in-degree stone,
it won't work on this "8" like graph:
[[1, 1, 0, 0, 0],
[1, 1, 0, 0, 0],
[0, 1, 1, 0, 0],
[0, 0, 1, 1, 1],
[0, 0, 0, 1, 1]]

The stone in the center has least degree = 2.
But if you remove this stone first,
the whole connected stones split into 2 parts,
and you will finish with 2 stones left.


3. A good strategy
In fact, the proof is really straightforward.
You probably apply a DFS, from one stone to next connected stone.
You can remove stones in reversed order.
In this way, all stones can be removed but the stone that you start your DFS.

One more step of explanation:
In the view of DFS, a graph is explored in the structure of a tree.
As we discussed previously,
a tree can be removed in topological order,
from leaves to root.


4. Count the number of islands
We call a connected graph as an island.
One island must have at least one stone left.
The maximum stones can be removed = stones number - islands number

The whole problem is transferred to:
What is the number of islands?

You can show all your skills on a DFS implementation,
and solve this problem as a normal one.


5. Unify index
Struggle between rows and cols?
You may duplicate your codes when you try to the same thing on rows and cols.
In fact, no logical difference between col index and rows index.

An easy trick is that, add 10000 to col index.
So we use 0 ~ 9999 for row index and 10000 ~ 19999 for col.


6. Search on the index, not the points
When we search on points,
we alternately change our view on a row and on a col.

We think:
a row index, connect two stones on this row
a col index, connect two stones on this col.

In another view：
A stone, connect a row index and col.

Have this idea in mind, the solution can be much simpler.
The number of islands of points,
is the same as the number of islands of indexes.


7. Union-Find
I use union find to solve this problem.
As I mentioned, the elements are not the points, but the indexes.

for each point, union two indexes.
return points number - union number
Copy a template of union-find,
write 2 lines above,
you can solve this problem in several minutes.


Complexity
union and find functions have worst case O(N), amortize O(1)
The whole union-find solution with path compression,
has O(N) Time, O(N) Space

If you have any doubts on time complexity,
please refer to wikipedia first.'''
# shorter version
class Solution(object):
    def removeStones(self, stones):
        """
        :type stones: List[List[int]]
        :rtype: int
        """
        uf = {}
        def find(x):
            if x != uf.setdefault(x, x):
                uf[x] = find(uf[x])
            return uf[x]
        for i, j in stones:
            uf[find(i)] = find(~j)
        return len(stones) - len({find(x) for x in uf})

class Solution(object):
    def removeStones(self, stones):
        """
        :type stones: List[List[int]]
        :rtype: int
        """
        UF = {}
        def find(x):
            if x != UF[x]:
                UF[x] = find(UF[x])
            return UF[x]
        def union(x, y):
            UF.setdefault(x, x)
            UF.setdefault(y, y)
            UF[find(x)] = find(y)

        for i, j in stones:
            union(i, ~j)
        return len(stones) - len({find(x) for x in UF})

'''
Update About Union Find Complexity
I have 3 main reasons that always insist O(N), on all my union find solutions.

The most important, union find is really a common knowledge for algorithm.
Using both path compression, splitting, or halving and union by rank or size ensures
that the amortized time per operation is only O(1).
So it's fair enough to apply this conclusion.

It's really not my job to discuss how union find works or the definition of big O.
I bet everyone can find better resource than my post on this part.
You can see the core of my solution is to transform the problem as a union find problem.
The essence is the thinking process behind.
People can have their own template and solve this problem with 2-3 more lines.
But not all the people get the point.

I personally manually write this version of union find every time.
It is really not worth a long template.
The version with path compression can well handle all cases on leetcode.
What‘s the benefit here to add more lines?

In this problem, there is N union operation, at most 2 * sqrt(N) node.
When N get bigger, the most operation of union operation is amortize O(1).

I knew there were three good resourse of union find:

top down analusis of path compression
wiki
stackexchange
But they most likely give a upper bound time complexity of union find,
not a supreme.
If anyone has a clear example of union find operation sequence,
to make it larger than O(N), I am so glad to know it.
'''
# Day 20 Brute Force/Backtracking
# 39. Combination Sum

# 46. Permutations