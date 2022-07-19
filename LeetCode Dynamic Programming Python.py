# Day 1
# 1137. N-th Tribonacci Number
# Solution: DP
# Time complexity: O(n)
# Space complexity: O(n) -> O(1)

class Solution(object):
    def tribonacci(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return n
        t0 = 0
        t1 = 1
        t2 = 1
        t = 1
        for i in range(3, n+1):
            t = t0+t1+t2
            t0 = t1
            t1 = t2
            t2 = t
        return t

# 509. Fibonacci Number

class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return n
        f0 = 0
        f1 = 1
        f = 1
        for i in range(2, n+1):
            f = f0+f1
            f0 = f1
            f1 = f
        return f

# Day 2
# 70. Climbing Stairs
# Solution: DP
# Time complexity: O(n)
# Space complexity: O(1)
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        one,two,curr=1,1,1
        for i in range(2,n+1):
            curr = one+two
            two = one
            one = curr
        return curr

class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        f = [0 for i in range(0,n+1)]
        f[0], f[1] = 1, 1
        for i in range(2,n+1):
            f[i] = f[i-1] + f[i-2]
            
        return f[n]
# 746. Min Cost Climbing Stairs
# O(1) Space
class Solution(object):
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        dp0,dp1=0,0
        for i in range(2,len(cost)+1):
            dp = min(dp1+cost[i-1],dp0+cost[i-2])
            dp0 = dp1
            dp1 = dp
        return dp
        
# Day 3
# 198. House Robber
# DP
# dp[i]: Max money after "visiting" house[i]
# dp[i] = max(dp[i-2] + money[i], dp[i-1])
#
# Time complexity: O(n)
# Space complexity: O(1)
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums: 
            return 0
        dp0 = 0
        dp1 = 0
        for i in range(0,len(nums)):
            dp = max(dp0 + nums[i], dp1)
            dp0 = dp1
            dp1 = dp
            
        return dp1

# 213. House Robber II
'''Now, what we have here is circular pattern.
Imagine, that we have 10 houses: a0, a1, a2, a3, ... a9: 
Then we have two possible options:

Rob house a0, then we can not rob a0 or a9 and we have a2, a3, ..., a8 range to rob
Do not rob house a0, then we have a1, a2, ... a9 range to rob.
Then we just choose maximum of these two options and we are done!

Complexity: time complexity is O(n), 
because we use dp problem with complexity O(n) twice. 
Space complexity is O(1), 
because in python lists passed by reference and space complexity of House Robber problem is O(1).
'''
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def rob_helper(nums):
            dp0 = 0
            dp1 = 0
            for num in nums:
                dp = max(dp0+num,dp1)
                dp0 = dp1
                dp1 = dp          
            return dp1
    
        return max(nums[0] + rob_helper(nums[2:-1]), rob_helper(nums[1:]))

# 740. Delete and Earn
# Counter
class Solution(object):
    def deleteAndEarn(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        points = Counter(nums)
        if not nums: 
            return 0
        dp0 = 0
        dp1 = 0
        for i in range(0, max(points.keys()) + 1):
            dp = max(dp0+i*points[i],dp1)
            dp0 = dp1
            dp1 = dp
            
        return dp1

# array
class Solution(object):
    def deleteAndEarn(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """       
        if not nums: 
            return 0
        
        points = [0] * (max(nums)+1)
        for num in nums:
            points[num] += num
        
        dp0 = 0
        dp1 = 0
        for i in range(len(points)):
            dp = max(dp0+points[i],dp1)
            dp0 = dp1
            dp1 = dp
            
        return dp1

# Day 4
# 55. Jump Game

'''
Complexity

Time: O(N), where N <= 10^4 is length of nums array.
Space: O(1)
'''
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        n = len(nums)
        maxPos = 0
        i = 0
        while i <= maxPos:
            maxPos = max(maxPos, i + nums[i])
            if maxPos >= n - 1: return True
            i += 1
        
        return False

class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        m = 0
        for i, n in enumerate(nums):
            if i > m:
                return False
            m = max(m, i+n)
        return True

# 45. Jump Game II
'''
Idea:
Since each element of our input array (N) represents the maximum jump length and not the definite jump length, 
that means we can visit any index between the current index (i) and i + N[i]. Stretching that to its logical conclusion, 
we can safely iterate through N while keeping track of the furthest index reachable (next) at any given moment (next = max(next, i + N[i])). 
We'll know we've found our solution once next reaches or passes the last index (next >= N.length - 1).

The difficulty then lies in keeping track of how many jumps it takes to reach that point. 
We can't simply count the number of times we update next, 
as we may see that happen more than once while still in the current jump's range. 
In fact, we can't be sure of the best next jump until we reach the end of the current jump's range.

So in addition to next, we'll also need to keep track of the current jump's endpoint (curr) 
as well as the number of jumps taken so far (ans).

Since we'll want to return ans at the earliest possibility, 
we should base it on next, as noted earlier. 
With careful initial definitions for curr and next, 
we can start our iteration at i = 0 and ans = 0 without the need for edge case return expressions.

Time Complexity: O(N) where N is the length of N
Space Cmplexity: O(1)
'''
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        curr, dp, ans, i =  -1, 0, 0, 0
        while dp < len(nums)-1:
            if i > curr:
                ans += 1
                curr = dp
            dp = max(dp, nums[i] + i)
            i += 1
        return ans

'''
Greedy

The main idea is based on greedy.
Step 1: Let's say the range of the current jump is [left, right],
 farthest is the farthest position that all positions in [left, right] can reach.
Step 2: Once we reach to right, we trigger another jump with left = right + 1,
 right = farthest, then repeat step 1 util we reach at the end.
 
Complexity
Time: O(N), where N <= 10^4 is the length of array nums
Space: O(1)
'''
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        jumps = 0
        dp = 0
        l = r = 0
        while r < len(nums) - 1:
            for i in range(l, r + 1):
                dp = max(dp, i + nums[i])
            l = r + 1
            r = dp
            jumps += 1
            
        return jumps

# Day 5
# 53. Maximum Subarray

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """    
        for i in range(1,len(nums)):
            if nums[i-1] >= 0:
                nums[i] += nums[i-1]
        return max(nums)

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """    
        sum, res = nums[0], nums[0]
        for i in range(1, len(nums)):
            sum = nums[i] if sum < 0 else sum + nums[i]
            res = max(res, sum)
        return res

# 918. Maximum Sum Circular Subarray
'''
Intuition
I guess you know how to solve max subarray sum (without circular).
If not, you can have a reference here: 53. Maximum Subarray

Explanation
So there are two case.
Case 1. The first is that the subarray take only a middle part, and we know how to find the max subarray sum.
Case2. The second is that the subarray take a part of head array and a part of tail array.
We can transfer this case to the first one.
The maximum result equals to the total sum minus the minimum subarray sum.

So the max subarray circular sum equals to
max(the max subarray sum, the total sum - the min subarray sum)

Prove of the second case
max(prefix+suffix)
= max(total sum - subarray)
= total sum + max(-subarray)
= total sum - min(subarray)

Corner case
Just one to pay attention:
If all numbers are negative, maxSum = max(A) and minSum = sum(A).
In this case, max(maxSum, total - minSum) = 0, which means the sum of an empty subarray.
According to the deacription, We need to return the max(A), instead of sum of am empty subarray.
So we return the maxSum to handle this corner case.

Complexity
One pass, time O(N)
No extra space, space O(1)
'''
class Solution(object):
    def maxSubarraySumCircular(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        total, maxSum, curMax, minSum, curMin = 0, nums[0], 0, nums[0], 0
        for a in nums:
            curMax = max(curMax + a, a)
            maxSum = max(maxSum, curMax)
            curMin = min(curMin + a, a)
            minSum = min(minSum, curMin)
            total += a
        return max(maxSum, total - minSum) if maxSum > 0 else maxSum

# Day 6
# 152. Maximum Product Subarray
class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        B = nums[::-1]
        for i in range(1, len(nums)):
            nums[i] *= nums[i - 1] or 1
            B[i] *= B[i - 1] or 1
        return max(nums + B)

# 1567. Maximum Length of Subarray With Positive Product
class Solution(object):
    def getMaxLen(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        pos, neg = 0, 0
        if nums[0] > 0: pos = 1
        if nums[0] < 0: neg = 1
        ans = pos
        for i in range(1, n):
            if nums[i] > 0:
                pos = 1 + pos
                neg = 1 + neg if neg > 0 else 0
            elif nums[i] < 0:
                pos, neg = 1 + neg if neg > 0 else 0, 1 + pos
            else:
                pos, neg = 0, 0
            ans = max(ans, pos)
        return ans

# Day 7
# 1014. Best Sightseeing Pair
'''
Soluton 1
Count the current best score in all previous sightseeing spot.
Note that, as we go further, the score of previous spot decrement.

cur will record the best score that we have met.
We iterate each value a in the array A,
update res by max(res, cur + a)

Also we can update cur by max(cur, a).
Note that when we move forward,
all sightseeing spot we have seen will be 1 distance further.

So for the next sightseeing spot cur = Math.max(cur, a) - 1

Complexity:
One pass,
Time O(N),
Space O(1).
'''
class Solution(object):
    def maxScoreSightseeingPair(self, values):
        """
        :type values: List[int]
        :rtype: int
        """
        cur = res = 0
        for a in values:
            res = max(res, cur + a)
            cur = max(cur, a) - 1
        return res

# solution 2
class Solution(object):
    def maxScoreSightseeingPair(self, values):
        """
        :type values: List[int]
        :rtype: int
        """
        res = imax = 0
        for i, a in enumerate(values):
            res = max(res, imax + values[i] - i)
            imax = max(imax, values[i] + i)
        return res

# 121. Best Time to Buy and Sell Stock
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices)<=1: 
            return 0
        maxi = 0     #the maximum profit you can get if you sell stock on the current day
        mini = prices[0]  #the minimum price you can buy stock before the current day
        for i in range(len(prices)):
            mini = min(mini,prices[i]) #try to update mini
            maxi = max(maxi,prices[i]-mini) #try to update maxi
        return maxi

# 122. Best Time to Buy and Sell Stock II
'''Solution 2: Greedy
Complexity:

Time: O(N)
Space: O(1)
'''
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        n = len(prices)
        maxProfit = 0
        for i in range(n-1):
            if prices[i+1] > prices[i]:
                maxProfit += prices[i+1] - prices[i]
        return maxProfit

# Day 8
# 309. Best Time to Buy and Sell Stock with Cooldown
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        sold,rest,hold = 0,0,float('-inf')
        for price in prices:
            prev = sold
            sold = hold+price
            hold = max(hold,rest-price)
            rest = max(rest,prev)
        return max(rest,sold)

# 714. Best Time to Buy and Sell Stock with Transaction Fee
class Solution(object):
    def maxProfit(self, prices, fee):
        """
        :type prices: List[int]
        :type fee: int
        :rtype: int
        """
        cash, hold = 0, -prices[0]
        for i in range(1, len(prices)):
            cash = max(cash, hold + prices[i] - fee)
            hold = max(hold, cash - prices[i])
        return cash
# Day 9
# 139. Word Break
# DP
# Time complexity O(n^2)
# Space complexity O(n^2)
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        def canBreak(s, m, wordDict):
            if s in m: return m[s]
            if s in wordDict: 
                m[s] = True
                return True
            
            for i in range(1, len(s)):
                r = s[i:]
                if r in wordDict and canBreak(s[0:i], m, wordDict):
                    m[s] = True
                    return True
            
            m[s] = False
            return False
            
        return canBreak(s, {}, set(wordDict))

# 42. Trapping Rain Water
# two-pointer
# Time complexity: O(n)
# Space complexity: O(1)
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if len(height) <= 0: return 0
        n = len(height)
        maxL, maxR = height[0], height[n-1]
        l, r = 0, n-1
        ans = 0
        while l < r:
            if maxL < maxR:
                ans += maxL - height[l]
                l += 1
                maxL = max(maxL, height[l])
            else:
                ans += maxR - height[r]
                r -= 1
                maxR = max(maxR, height[r])
        return ans

class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if len(height) <= 2: return 0
        n = len(height)
        maxLeft, maxRight = height[0], height[n-1]
        left, right = 1, n - 2
        ans = 0
        while left <= right:
            if maxLeft < maxRight:
                if height[left] > maxLeft:
                    maxLeft = height[left]
                else:
                    ans += maxLeft - height[left]
                left += 1
            else:
                if height[right] > maxRight:
                    maxRight = height[right]
                else:
                    ans += maxRight - height[right]
                right -= 1
        return ans

# DP
# l[i] := max(h[0:i+1])
# r[i] := max(h[i:n])
# ans = sum(min(l[i], r[i]) – h[i])
# Time complexity: O(n)
# Space complexity: O(n)
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        n = len(height)
        l, r = [0] * n, [0] * n
        ans = 0
        
        for i in range(0, n):
            if i == 0: l[i] = height[i]
            else: l[i] =  max(height[i], l[i-1])
        for i in range(n-1, -1, -1):
            if i == n-1: r[i] = height[i]
            else:r[i] = max(height[i], r[i+1])
            
        for i in range(n):
            waterLevel = min(l[i], r[i])
            if waterLevel >= height[i]:
                ans += waterLevel - height[i]
        return ans

class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        n = len(height)
        maxLeft, maxRight = [0] * n, [0] * n
        
        for i in range(1, n):
            maxLeft[i] = max(height[i-1], maxLeft[i-1])
        for i in range(n-2, -1, -1):
            maxRight[i] = max(height[i+1], maxRight[i+1])
            
        ans = 0
        for i in range(n):
            waterLevel = min(maxLeft[i], maxRight[i])
            if waterLevel >= height[i]:
                ans += waterLevel - height[i]
        return ans

# Day 10
# 413. Arithmetic Slices
class Solution(object):
    def numberOfArithmeticSlices(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        dp, dpPrev = 0, 0
        ans = 0
        for i in range(2, n):
            if nums[i-1] - nums[i-2] == nums[i] - nums[i-1]:
                dp = dpPrev + 1
            ans += dp
            dpPrev = dp
            dp = 0
        return ans
        
# 91. Decode Ways

class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        dp, dp1, dp2, n = 0, 1, 0, len(s)
        for i in range(n-1, -1, -1):
            if s[i] != '0':  # Single digit
                dp += dp1
            if i+1 < n and (s[i] == '1' or s[i] == '2' and s[i+1] <= '6'):  # Two digits
                dp += dp2
            dp, dp1, dp2 = 0, dp, dp1
        return dp1

# Day 11
# 264. Ugly Number II
class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        factors, k = [2,3,5], 3
        starts, Numbers = [0] * k, [1]
        for i in range(n-1):
            candidates = [factors[i]*Numbers[starts[i]] for i in range(k)]
            new_num = min(candidates)
            Numbers.append(new_num)
            starts = [starts[i] + (candidates[i] == new_num) for i in range(k)]
        return Numbers[-1]

class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        nums, idx, factors = [1], [0,0,0], [2, 3, 5]
        for _ in range(n-1):
            cans = [x * y for x,y in zip([nums[i] for i in idx], factors)]
            nums += min(cans),
            idx = [i + (c == nums[-1]) for i,c in zip(idx, cans)]
            
        return nums[-1]
# 96. Unique Binary Search Trees
# DP
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        res = [0] * (n+1)
        res[0] = 1
        for i in xrange(1, n+1):
            for j in xrange(i):
                res[i] += res[j] * res[i-1-j]
        return res[n]

# Catalan Number  (2n)!/((n+1)!*n!)  
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        return math.factorial(2*n)/(math.factorial(n)*math.factorial(n+1))

# Day 12
# 118. Pascal's Triangle
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        res = [[1]]
        for i in range(1, numRows):
            res += [map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1])]
        return res[:numRows]

# 119. Pascal's Triangle II
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        ans = [1]*(rowIndex+1)
        up = rowIndex
        down = 1
        for i in range(1, rowIndex):
            ans[i] = ans[i-1]*up/down
            up = up - 1
            down = down + 1
        return ans

# Day 13
# 931. Minimum Falling Path Sum
class Solution(object):
    def minFallingPathSum(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        n, m = len(matrix), len(matrix[0])
        
        for i in range(1, n):
            for j in range(m):
                s = matrix[i - 1][j]
                if j > 0: s = min(s, matrix[i - 1][j - 1])
                if j < m - 1: s = min(s, matrix[i - 1][j + 1])
                matrix[i][j] += s
          
        import numpy
        return numpy.min(matrix[-1])
        # return sorted(matrix[-1])[0] works too less memory more time
# 120. Triangle
'''
Solution: DP
Time complexity: O(n^2)
Space complexity: O(1)'''
class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        n = len(triangle)
        for i in range(n):
            for j in range(i+1):
                if i==0 and j==0: continue
                elif j==0: triangle[i][j] += triangle[i-1][j]
                elif j==i: triangle[i][j] += triangle[i-1][j-1]
                else: triangle[i][j] += min(triangle[i-1][j],triangle[i-1][j-1])
        return sorted(triangle[-1])[0]
        # import numpy
        # return numpy.min(triangle[-1]) works too less memory more time
# Day 14
# 1314. Matrix Block Sum
'''
rangeSum[i + 1][j + 1] corresponds to cell (i, j);
rangeSum[0][j] and rangeSum[i][0] are all dummy values, 
which are used for the convenience of computation of DP state transmission formula.
Analysis:
Time & space: O(m * n).
'''
class Solution(object):
    def matrixBlockSum(self, mat, k):
        """
        :type mat: List[List[int]]
        :type k: int
        :rtype: List[List[int]]
        """
        m, n = len(mat), len(mat[0])
        rangeSum = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                rangeSum[i + 1][j + 1] = rangeSum[i + 1][j] + rangeSum[i][j + 1] - rangeSum[i][j] + mat[i][j]
        ans = [[0] * n for _ in range(m)]        
        for i in range(m):
            for j in range(n):
                r1, c1, r2, c2 = max(0, i - k), max(0, j - k), min(m, i + k + 1), min(n, j + k + 1)
                ans[i][j] = rangeSum[r2][c2] - rangeSum[r1][c2] - rangeSum[r2][c1] + rangeSum[r1][c1]
        return ans

# 304. Range Sum Query 2D - Immutable
'''
Complexity
Constructor: Time & Space: O(m*n), where m is the number of rows, n is the number of columns in the grid
sumRegion: Time & Space: O(1)
'''
class NumMatrix(object):

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        m, n = len(matrix), len(matrix[0])
        self.sum = [[0] * (n + 1) for _ in range(m + 1)]  # sum[i][j] is sum of all elements inside the rectangle [0,0,i,j]
        for r in range(1, m + 1):
            for c in range(1, n + 1):
                self.sum[r][c] = self.sum[r - 1][c] + self.sum[r][c - 1] - self.sum[r - 1][c - 1] + matrix[r - 1][c - 1]


    def sumRegion(self, r1, c1, r2, c2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        r1, c1, r2, c2 = r1 + 1, c1 + 1, r2 + 1, c2 + 1  # Since our `sum` starts by 1 so we need to increase r1, c1, r2, c2 by 1
        return self.sum[r2][c2] - self.sum[r2][c1 - 1] - self.sum[r1 - 1][c2] + self.sum[r1 - 1][c1 - 1]


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)

# Day 15
# 62. Unique Paths
'''
Solution : Math Solution
There are total m+n-2 moves to go from Top-Left to Bottom-Right.
In m+n-2 moves, there are m-1 down moves and n-1 right moves.
You can imagine there are m+n-2 moves as: X X X ... X X X
X can be one of two values: down D or right R.
So, basically, it means we need to calculate how many ways we could choose m-1 down moves from m+n-2 moves, or n-1 right moves from m+n-2 moves.
So total ways = C(m+n-2, m-1) = C(m+n-2, n-1) = (m+n-2)! / (m-1)! / (n-1)!.

Complexity
Time: O(M + N), where M <= 100 is number of rows, N <= 100 is number of columns.
Space: O(1)
'''
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        ans = 1
        j = 1
        for i in range(m, m+n-2 + 1):
            ans *= i
            ans #= j
            j += 1
            
        return ans

# O(n) space 
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        if not m or not n:
            return 0
        cur = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                cur[j] += cur[j-1]
        return cur[-1]

# 63. Unique Paths II
# DP In-place
'''
Complexity Analysis
Time Complexity: O(M times N)O(M×N). 
The rectangular grid given to us is of size M times N and we process each cell just once.
Space Complexity: O(1). 
We are utilizing the obstacleGrid as the DP array. Hence, no extra space.'''
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])

        # If the starting cell has an obstacle, then simply return as there would be
        # no paths to the destination.
        if obstacleGrid[0][0] == 1:
            return 0

        # Number of ways of reaching the starting cell = 1.
        obstacleGrid[0][0] = 1

        # Filling the values for the first column
        for i in range(1,m):
            obstacleGrid[i][0] = int(obstacleGrid[i][0] == 0 and obstacleGrid[i-1][0] == 1)

        # Filling the values for the first row        
        for j in range(1, n):
            obstacleGrid[0][j] = int(obstacleGrid[0][j] == 0 and obstacleGrid[0][j-1] == 1)

        # Starting from cell(1,1) fill up the values
        # No. of ways of reaching cell[i][j] = cell[i - 1][j] + cell[i][j - 1]
        # i.e. From above and left.
        for i in range(1,m):
            for j in range(1,n):
                if obstacleGrid[i][j] == 0:
                    obstacleGrid[i][j] = obstacleGrid[i-1][j] + obstacleGrid[i][j-1]
                else:
                    obstacleGrid[i][j] = 0

        # Return value stored in rightmost bottommost cell. That is the destination.            
        return obstacleGrid[m-1][n-1]

# in place
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        if not obstacleGrid:
            return 
        r, c = len(obstacleGrid), len(obstacleGrid[0])
        obstacleGrid[0][0] = 1 - obstacleGrid[0][0]
        for i in xrange(1, r):
            obstacleGrid[i][0] = obstacleGrid[i-1][0] * (1 - obstacleGrid[i][0])
        for i in xrange(1, c):
            obstacleGrid[0][i] = obstacleGrid[0][i-1] * (1 - obstacleGrid[0][i])
        for i in xrange(1, r):
            for j in xrange(1, c):
                obstacleGrid[i][j] = (obstacleGrid[i-1][j] + obstacleGrid[i][j-1]) * (1 - obstacleGrid[i][j])
        return obstacleGrid[-1][-1]

# O(n) space
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        if not obstacleGrid:
            return 
        r, c = len(obstacleGrid), len(obstacleGrid[0])
        cur = [0] * c
        cur[0] = 1 - obstacleGrid[0][0]
        for i in xrange(1, c):
            cur[i] = cur[i-1] * (1 - obstacleGrid[0][i])
        for i in xrange(1, r):
            cur[0] *= (1 - obstacleGrid[i][0])
            for j in xrange(1, c):
                cur[j] = (cur[j-1] + cur[j]) * (1 - obstacleGrid[i][j])
        return cur[-1]
# O(m*n) space
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        if not obstacleGrid:
            return 
        r, c = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0 for _ in xrange(c)] for _ in xrange(r)]
        dp[0][0] = 1 - obstacleGrid[0][0]
        for i in xrange(1, r):
            dp[i][0] = dp[i-1][0] * (1 - obstacleGrid[i][0])
        for i in xrange(1, c):
            dp[0][i] = dp[0][i-1] * (1 - obstacleGrid[0][i])
        for i in xrange(1, r):
            for j in xrange(1, c):
                dp[i][j] = (dp[i][j-1] + dp[i-1][j]) * (1 - obstacleGrid[i][j])
        return dp[-1][-1]

# Day 16
# 64. Minimum Path Sum
''' Bottom up DP - In place
Idea
For the current cell [r, c], there are two options to choose:
Choose from up cell [r-1, c] go down to [r, c].
Choose from left cell [r, c-1] go right to [r, c].
So, we need to choose minium cost between 2 above options.

Complexity:
Time: O(M*N), where M <= 200 is number of rows, N <= 200 is number of columns.
Space: O(1)'''
class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m, n = len(grid), len(grid[0])
        for r in range(m):
            for c in range(n):
                if r == 0 and c == 0:
                    pass
                elif r == 0:
                    grid[r][c] += grid[r][c-1]
                elif c == 0:
                    grid[r][c] += grid[r-1][c]
                else:
                    grid[r][c] += min(grid[r-1][c], grid[r][c-1])
        return grid[m-1][n-1]

# 221. Maximal Square
'''
Top down DP 
Let dp[r][c] denote the side length of the maximum square whose bottom right corner is the at cell (r, c).

Complexity:
Time: O(M*N), where M <= 300 is number of rows, N <= 300 is number of columns in the matrix.
Space: O(M*N)
'''
class Solution(object):
    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * n for _ in range(m)]
        ans = 0
        for r in range(m):
            for c in range(n):
                if matrix[r][c] == "0": continue
                if r == 0 or c == 0:
                    dp[r][c] = 1
                else:
                    dp[r][c] = min(dp[r-1][c], dp[r][c-1], dp[r-1][c-1]) + 1
                ans = max(ans, dp[r][c])
        return ans * ans
# O(2*n) space 
class Solution(object):
    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if not matrix:
            return 0
        r, c = len(matrix), len(matrix[0])
        pre = cur = [0] * (c+1)
        res = 0
        for i in xrange(r):
            for j in xrange(c):
                cur[j+1] = (min(pre[j], pre[j+1], cur[j])+1)*int(matrix[i][j])
                res = max(res, cur[j+1]**2)
            pre = cur
            cur = [0] * (c+1)
        return res
# O(m*n) space, one pass 
class Solution(object):
    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if not matrix:
            return 0
        r, c = len(matrix), len(matrix[0])
        dp = [[int(matrix[i][j]) for j in xrange(c)] for i in xrange(r)]
        res = max(max(dp))
        for i in xrange(1, r):
            for j in xrange(1, c):
                dp[i][j] = (min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j])+1)*int(matrix[i][j])
                res = max(res, dp[i][j]**2)
        return res
        
# Day 17
# 5. Longest Palindromic Substring
'''Approach : Expand Around Center

Complexity Analysis
Time complexity : O(n^2). Since expanding a palindrome around its center could take O(n) time, 
Space complexity : O(1).'''
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        res = ""
        for i in xrange(len(s)):
            # odd case, like "aba"
            tmp = self.helper(s, i, i)
            if len(tmp) > len(res):
                res = tmp
            # even case, like "abba"
            tmp = self.helper(s, i, i+1)
            if len(tmp) > len(res):
                res = tmp
        return res

    # get the longest palindrome, l, r are the middle indexes   
    # from inner to outer
    def helper(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return s[l+1:r]
# To make it short, we can use "max"
# It much slower
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        res = ""
        for i in range(len(s)):
            res = max(self.helper(s,i,i), self.helper(s,i,i+1), res, key=len)

        return res
       
        
    def helper(self,s,l,r):
        
        while 0<=l and r < len(s) and s[l]==s[r]:
                l-=1; r+=1
        return s[l+1:r]

# 516. Longest Palindromic Subsequence
# Bottom up DP (Space Optimized), you need only one dp array
# Complexity
# Time: O(N^2), where N <= 1000 is length of string s.
# Space: O(N)
class Solution(object):
    def longestPalindromeSubseq(self, s):
        """
        :type s: str
        :rtype: int
        """
        n = len(s)
        dp = [0] * n
        for i in range(n - 1, -1, -1):
            dp[i] = 1
            temp = 0
            for j in range(i + 1, n):
                pre = dp[j]
                if s[i] == s[j]:
                    dp[j] = temp + 2
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                temp = pre
        return dp[n - 1]
'''
Solution 2: Bottom up DP

Let dp[l][r] denote the length of the longest palindromic subsequence of s[l..r].
There are 2 options:
If s[l] == s[r] then dp[l][r] = dp[l+1][r-1] + 2
Elif s[l] != s[r] then dp[l][r] = max(dp[l+1][r], dp[l][r-1]).
Then dp[0][n-1] is our result.
Complexity
Time: O(N^2), where N <= 1000 is length of string s.
Space: O(N^2)
'''
class Solution(object):
    def longestPalindromeSubseq(self, s):
        """
        :type s: str
        :rtype: int
        """
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            dp[i][i] = 1
            for j in range(i+1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

        return dp[0][n - 1]
# Day 18
# 300. Longest Increasing Subsequence
'''
Solution 2: DP + Binary Search / Patience Sort

dp[i] := smallest tailing number of a increasing subsequence of length i + 1.

dp is an increasing array, we can use binary search to find the index to insert/update the array.

ans = len(dp)

Time complexity: O(nlogn)
Space complexity: O(n)
'''
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        d = []
        for x in nums:
            i = bisect_left(d, x)
            if i == len(d): 
                d.append(x)
            else:
                d[i] = x
        return len(d)

# Binary Search / Patience sorting
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        tails = [0] * len(nums)
        size = 0
        for x in nums:
            i, j = 0, size
            while i != j:
                m = (i + j) / 2
                if tails[m] < x:
                    i = m + 1
                else:
                    j = m
            tails[i] = x
            size = max(i + 1, size)
        return size

# Time Complexity: O(n^2) very slow
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        if len(nums) == 0: return 0
        n = len(nums)
        f = [1] * n
        for i in range(1, n):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    f[i] = max(f[i], f[j] + 1)
        return max(f)

# 376. Wiggle Subsequence
# So if an element is bigger than a previous element,
# it will only add to the count if there is one that is smaller between them,
# and vice versa.
# O(n) time O(1) space
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        
        length = 1
        up = None # current is increasing or not
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1] and up != True:
                length += 1
                up = True
            if nums[i] < nums[i - 1] and up != False:
                length += 1
                up = False
        return length

# Approach # Space-Optimized Dynamic Programming
# Complexity Analysis
# Time complexity : O(n). Only one pass over the array length.
# Space complexity : O(1). Constant space is used.
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2: return len(nums)
        
        down = 1
        up = 1 
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1] :
                up = down + 1
            if nums[i] < nums[i - 1] :
                down = up + 1
                
        return max(down, up)

# only in python 3 @lru_cache works in python 3
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        @lru_cache(None)
        def dp(i, s):
            if i == 0: return 1
            return dp(i-1, -s) + 1 if (nums[i] - nums[i-1])*s < 0 else dp(i-1, s)
            
        return max(dp(len(nums)-1, -1), dp(len(nums)-1, 1))

# Day 19
# 392. Is Subsequence
'''
Complexity

Time: O(M + N), where M <= 100 is length of s string, N <= 10^4 is length of t string.
Space: O(1)
'''
class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        i = 0
        for c in t:
            if i == len(s): return True  # If match full s -> then s is a subsequence
            if s[i] == c:
                i += 1
        return i == len(s)
'''
Follow up: Suppose there are lots of incoming s, say s1, s2, ..., sk where k >= 10^9, and you want to check one by one to see if t has its subsequence. In this scenario, how would you change your code?

Since we use t multiple times, we can pre-compute t string to jump to the next index faster.
The following code, I modify to assume we process k string s1, s2,..., sk, where |s| <= 100, |t| <= 10^4.

Complexity:
Time: O(K * M * logN + N), where K is the number of s strings, M <= 100 is length of each s string, N <= 10^4 is length of t string.
Space: O(N)
'''
class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        mapIndices = defaultdict(list)
        for i, c in enumerate(t):
            mapIndices[c].append(i)
            
        def findNextIndex(arr, startIdx):  # Find next index which start from `startIdx`
            # For example: [1, 3, 4, 5], startIdx = 3 -> Expect: idx = 1
            # For example: [1, 3, 4, 5], startIdx = 2 -> Expect: idx = 1
            # For example: [1, 3, 4, 5], startIdx = 0 -> Expect: idx = 0
            # For example: [1, 3, 4, 5], startIdx = 6 -> Expect: idx = 4
            idx = bisect_left(arr, startIdx)
            if idx == len(arr): return -1
            return arr[idx] + 1
            
        def isSubsequence(s, t):
            nextIdx = 0
            for i, c in enumerate(s):
                if nextIdx == len(t):
                    return False
                nextIdx = findNextIndex(mapIndices[c], nextIdx)
                if nextIdx == -1: return False
            return True
                    
        k = 10000  # Assume we process k string s1, s2,..., sk, where |s| <= 100
        ans = False
        for _ in range(k):
            ans = isSubsequence(s, t)
        return ans

# 1143. Longest Common Subsequence
'''
DP

Use dp[i][j] to represent the length of longest common sub-sequence of text1[0:i] and text2[0:j]
dp[i][j] = dp[i - 1][j - 1] + 1 if text1[i - 1] == text2[j - 1] else max(dp[i][j - 1], dp[i - 1][j])

Time complexity: O(mn)
Space complexity: O(mn) -> O(n)
'''
class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        """
        :type text1: str
        :type text2: str
        :rtype: int
        """
        m,n = len(text1),len(text2)    
        dp1,dp2 = [0]*(n+1), [0]*(n+1)     
        for i in range(m):
            for j in range(n) :        
                if text1[i] == text2[j]:
                    dp2[j + 1] = dp1[j] + 1 
                else:
                    dp2[j + 1] = max(dp1[j + 1] , dp2[j])
            dp1,dp2 = dp2,dp1
              
        return dp1[n] 

class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        """
        :type text1: str
        :type text2: str
        :rtype: int
        """
        m,n = len(text1),len(text2)    
        dp = [0]*(n+1)    
        for i in range(m):
            prev = 0; # dp[i][j]
            for j in range(n) :        
                curr = dp[j + 1] # dp[i][j + 1]
                if text1[i] == text2[j]:
                  # dp[i + 1][j + 1] = dp[i][j] + 1
                    dp[j + 1] = prev + 1 
                else:
                  # dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
                    dp[j + 1] = max(curr, dp[j])
                prev = curr
               
        return dp[n] # dp[m][n]
        
# 72. Edit Distance

# Solution : Bottom up DP (Space Optimized)
# Since we build our dp rows by rows, we access only previous dp state dpPrev and current dp state dp.
# So we can optimize to O(N) in Space Complexity.
# Complexity:
# Time: O(M*N), where M <= 500 is length of s1 string, N <= 500 is length of s2 string.
# Space: O(N)
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m, n = len(word1), len(word2)
        dp, dpPrev = [-1] * (n+1), [-1] * (n+1)
        for i in range(m+1):
            for j in range(n+1):
                if i == 0:
                    dp[j] = j  # Need to insert `j` chars to become s2[:j]
                elif j == 0:
                    dp[j] = i  # Need to delete `i` chars to become ""
                elif word1[i-1] == word2[j-1]:
                    dp[j] = dpPrev[j-1]
                else:
                    dp[j] = min(dpPrev[j], dp[j-1], dpPrev[j-1]) + 1
            dp, dpPrev = dpPrev, dp
        return dpPrev[n]

# Solution : Bottom up DP
# Complexity:
# Time: O(M*N), where M <= 500 is length of s1 string, N <= 500 is length of s2 string.
# Space: O(M*N)
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m, n = len(word1), len(word2)
        dp = [[-1] * (n+1) for _ in range(m+1)]
        for i in range(m+1):
            for j in range(n+1):
                if i == 0:
                    dp[i][j] = j  # Need to insert `j` chars to become word2[:j]
                elif j == 0:
                    dp[i][j] = i  # Need to delete `i` chars to become ""
                elif word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        return dp[m][n]

# Iterative DP
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m, n = len(word1), len(word2)
        dp = [[0] * (n+1) for _ in range(m+1)]
        for i in range(m+1):
            for j in range(n+1):
                if i == 0:
                    dp[i][j] = j  # Need to insert `j` chars to become word2[:j]
                elif j == 0:
                    dp[i][j] = i  # Need to delete `i` chars to become ""
                if i >= 1 and j >= 1:    
                    if word1[i-1] == word2[j-1]:
                        c = 0
                    else:
                        c = 1
                    dp[i][j] = min(min(dp[i-1][j], dp[i][j-1])+1, dp[i-1][j-1] + c)
        return dp[m][n]

# Recursive DP super quick
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        l1, l2 = len(word1), len(word2)
        dp = [[-1] * (l2+1) for _ in range(l1+1)]
        return self.MinDistance(dp,word1,word2,l1,l2)
     
    def MinDistance(self,dp, word1, word2, l1, l2):
        if not l1 : return l2
        if not l2 : return l1
        if dp[l1][l2] >= 0 : return dp[l1][l2]

        ans = 0
        if word1[l1 - 1] == word2[l2 - 1]:
            ans = self.MinDistance(dp, word1, word2, l1 - 1, l2 - 1)
        else :
            ans = min(self.MinDistance(dp, word1, word2, l1 - 1, l2 - 1),
                  min(self.MinDistance(dp, word1, word2, l1 - 1, l2), 
                      self.MinDistance(dp, word1, word2, l1, l2 - 1))) + 1
        dp[l1][l2] = ans  
        return ans    

# Recursive DP similar with above, second quick    
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        l1, l2 = len(word1), len(word2)
        dp = [[-1] * (l2+1) for _ in range(l1+1)]
         
        def minDistance(word1, word2, l1, l2):
            if not l1 : return l2
            if not l2 : return l1
            if dp[l1][l2] >= 0 : return dp[l1][l2]

            ans = 0
            if word1[l1 - 1] == word2[l2 - 1]:
                ans = minDistance(word1, word2, l1 - 1, l2 - 1)
            else :
                ans = min(minDistance(word1, word2, l1 - 1, l2 - 1),
                      min(minDistance(word1, word2, l1 - 1, l2), 
                          minDistance(word1, word2, l1, l2 - 1))) + 1
            dp[l1][l2] = ans  
            return ans    
        
        return minDistance(word1,word2,l1,l2)
        
# Day 20
# 322. Coin Change
'''Knapsack problem
This is a classic knapsack problem. Honestly, I'm not good at knapsack problem, it's really tough for me.

dp[i][j] : the number of combinations to make up amount j by using the first i types of coins
State transition:

not using the ith coin, only using the first i-1 coins to make up amount j, then we have dp[i-1][j] ways.
using the ith coin, since we can use unlimited same coin, we need to know how many ways to make up amount j - coins[i-1] by using first i coins(including ith), which is dp[i][j-coins[i-1]]
Initialization: dp[i][0] = 1

Once you figure out all these, it's easy to write out the code:
'''
class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        dp = [0]*(amount+1)
        dp[0] = 1
        for coin in coins:
            for i in range(coin, amount+1) :
                dp[i] += dp[i-coin]
            
        
        return dp[amount]

# 518. Coin Change 2
'''Solution : DP
Time complexity: O(n*amount)
Space complexity: O(amount)
'''
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        INVALID = 2**32
        dp = [INVALID] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for i in range(coin, amount + 1):
                if dp[i - coin] >= dp[i]: continue
                dp[i] = dp[i - coin] + 1
        return -1 if dp[amount] == INVALID else dp[amount]
'''
DFS unfortunely is TLE
Time complexity: O(amount^n/(coin_0*coin_1*…*coin_n))
Space complexity: O(n)
'''
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        coins.sort(reverse=True)
        INVALID = 10**10
        self.ans = INVALID
        def dfs(s, amount, count):      
            if amount == 0:
                self.ans = count
                return
            if s == len(coins): return

            coin = coins[s]
            for k in range(amount // coin, -1, -1):
                if count + k >= self.ans: break
                dfs(s + 1, amount - k * coin, count + k)
        dfs(0, amount, 0)
        return -1 if self.ans == INVALID else self.ans
# Day 21
# 377. Combination Sum IV
# DP
class Solution(object):
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        dp = [0]*(target + 1)
        dp[0] = 1
        for i in range(1,target+1):
            for num in nums:
                if i - num >= 0:
                    dp[i] += dp[i - num]           
        return dp[target]

# 343. Integer Break
class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
    
        if n > 3: n+=1
        dp = [0]*(n+1)
        dp[1] = 1
       # fill the entire dp array
        for i in range(2,n+1):
            for j in range(1,i):
                dp[i] = max(dp[i], j*dp[i - j])
        return dp[n]

# 279. Perfect Squares
'''Solution : DP
dp[i] := ans
dp[0] = 0
dp[i] = min{dp[i – j * j] + 1} 1 <= j * j <= i

dp[5] = min{
dp[5 – 2 * 2] + 1 = dp[1] + 1 = (dp[1 – 1 * 1] + 1) + 1 = dp[0] + 1 + 1 = 2,
dp[5 – 1 * 1] + 1 = dp[3] + 1 = (dp[3 – 1 * 1] + 1) + 1 = dp[1] + 2 = dp[1 – 1*1] + 1 + 2 = dp[0] + 3 = 3
};

dp[5] = 2

Time complexity: O(n * sqrt(n))
Space complexity: O(n)
'''
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [sys.maxint >> 1] * (n + 1) 
        
        dp[0] = 0
        for i in range(1, n+1):
            for j in range(1, int(i**0.5)+1):
                dp[i] = min(dp[i], dp[i - j * j] + 1)
        return dp[n]
    