// Day 1
// 1137. N-th Tribonacci Number
// Solution: DP
// Time complexity: O(n)
// Space complexity: O(n) -> O(1)
public class Solution {
    public int Tribonacci(int n) {
        if(n == 0)
        {
            return n;
        }
        int t0 = 0;
        int t1 = 1;
        int t2 = 1;
        int t = 1;
        for(int i = 3; i < n+1; i++ )
        {
            t = t0+t1+t2;
            t0 = t1;
            t1 = t2;
            t2 = t;
        }
        return t;
    }
}

// 509. Fibonacci Number
public class Solution {
    public int Fib(int n) {
        if(n == 0)
        {
            return n;
        }
        int f0 = 0;
        int f1 = 1;
        int f = 1;
        for(int i = 2; i < n+1; i++ )
        {
            f = f0+f1;
            f0 = f1;
            f1 = f;
        }
        return f;
    }
}

// Day 2
// 70. Climbing Stairs
//Solution: DP
//Time complexity: O(n)
//Space complexity: O(1)
public class Solution {
    public int ClimbStairs(int n) {
        int zero = 1, one = 1, curr = 1;
        for(int i = 2; i<=n;i++)
        {
            curr = zero + one;
            one = zero;
            zero = curr;
        }
        return curr;
    }
}

//Solution: DP
//Time complexity: O(n)
//Space complexity: O(n)
public class Solution {
    public int ClimbStairs(int n) {
        // f[i] = climbStairs(i)
        int[] f = new int[n+1];
        Array.Fill(f, 0);
        f[0] = f[1] = 1;
        // f[i] = f[i-1] + f[i-2]
        for (int i = 2;i <= n; ++i)
          f[i] = f[i - 1] + f[i - 2];
        return f[n];
    }
}

// 746. Min Cost Climbing Stairs
// O(1) Space
public class Solution {
    public int MinCostClimbingStairs(int[] cost) {
        int dp0 = 0;
        int dp1 = 0;
        for (int i = 2; i <= cost.Length; i++) {
            int dp = Math.Min(dp1 + cost[i - 1], dp0 + cost[i - 2]);
            dp0 = dp1;
            dp1 = dp;
        }
        return dp1; 
    }
}

// O(n) Space
public class Solution {
    public int MinCostClimbingStairs(int[] cost) {
        int n = cost.Length;
        int[] dp = new int[n]; // d[i] min cost before leaving i
        Array.Fill(dp, Int32.MaxValue);
        dp[0] = cost[0];
        dp[1] = cost[1];
        for (int i = 2; i < n; ++i)
            dp[i] = Math.Min(dp[i - 1], dp[i - 2]) + cost[i];
        // We can reach top from either n - 1, or n - 2
        return Math.Min(dp[n - 1], dp[n - 2]);
    }
}

// Day 3
// 198. House Robber
// DP
// Time complexity: O(n)
// Space complexity: O(1)
public class Solution {
    public int Rob(int[] nums) {
        if (nums.Length == 0) return 0;
        int dp0 = 0;
        int dp1 = 0;
        for (int i = 0; i < nums.Length;i++) {
            int dp = Math.Max(dp0 + nums[i], dp1);
            dp0 = dp1;
            dp1 = dp;
        }
        return dp1;
    }
}

// 213. House Robber II
public class Solution {
    public int Rob(int[] nums) {
        if(nums.Length<1) {return 0;}
        else if(nums.Length ==1) { return nums[0];}
        else
        {
            return Math.Max(RobHelp(nums[0..^1]),RobHelp(nums[1..^0]));
        }
        
    }
    
    public int RobHelp(int[] nums)
    {   if (nums.Length == 0) return 0;
        int dp0 = 0;
        int dp1 = 0;
        for (int i = 0; i < nums.Length;i++) {
            int dp = Math.Max(dp0 + nums[i], dp1);
            dp0 = dp1;
            dp1 = dp;
        }
        return dp1;
    }
}

public class Solution {
    public int Rob(int[] nums) {
        if (nums.Length == 0) return 0;
        if (nums.Length == 1) return nums[0];
        return Math.Max(Rob(nums, 0, nums.Length - 2), Rob(nums, 1, nums.Length - 1));
    }
    
    public int Rob(int[] nums, int start, int end) 
    {   int dp0 = 0;
        int dp1 = 0;
        int dp = 0;
        for (int i = start; i <= end; i++)
        {
            dp = Math.Max(dp0 + nums[i], dp1);
            dp0 = dp1;
            dp1 = dp;
        }
        return dp;
    }
}

// 740. Delete and Earn
/*
Key observations: If we take nums[i]

We can safely take all of its copies.
We can’t take any of copies of nums[i – 1] and nums[i + 1]
This problem is reduced to 198 House Robber.

Houses[i] has all the copies of num whose value is i.

[3 4 2] -> [0 2 3 4], rob([0 2 3 4]) = 6            

[2, 2, 3, 3, 3, 4] -> [0 2*2 3*3 4], rob([0 2*2 3*3 4]) = 9

Time complexity: O(n+r) reduction + O(r) solving rob = O(n + r)

Space complexity: O(r)

r = max(nums) – min(nums) + 1
*/
public class Solution {
    public int DeleteAndEarn(int[] nums) {
        if(nums.Length == 0)  return 0;
        
        int[] points = new int[nums.Max()+1]; //nums.Max()+1 is enough no need for 10001 everytime
        for (int i = 0; i<nums.Length;i++)
        {
            points[nums[i]] += nums[i];
        }
            
        return RobHelp(points);
        
    }
    
    public int RobHelp(int[] nums)
    {   if (nums.Length == 0) return 0;
        int dp0 = 0;
        int dp1 = 0;
        for (int i = 0; i < nums.Length;i++) {
            int dp = Math.Max(dp0 + nums[i], dp1);
            dp0 = dp1;
            dp1 = dp;
        }
        return dp1;
    }
}


// Day 4
// 55. Jump Game
// Max Pos So Far
// Complexity
// Time: O(N), where N <= 10^4 is length of nums array.
// Space: O(1)
public class Solution {
    public bool CanJump(int[] nums) {
        int n = nums.Length, dp = 0,i = 0;
        while(i <= dp){// if previous maxLocation dp smaller than i, meaning we cannot reach location i, thus return false.
            dp = Math.Max(dp, i + nums[i]);
            if (dp >= n - 1) return true;
            i += 1;
        }
        
        return false;
    }
}

public class Solution {
    public bool CanJump(int[] nums) {
        int maxLocation = 0;
        for(int i=0; i<nums.Length; i++) {
            if(maxLocation<i) return false; // if previous maxLocation smaller than i, meaning we cannot reach location i, thus return false.
            maxLocation = (i+nums[i]) > maxLocation ? i+nums[i] : maxLocation; // greedy:
        }
        return true;

    }
}

// 45. Jump Game II
/*Idea:
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
*/
public class Solution {
    public int Jump(int[] nums) {
        int curr = -1, dp = 0, ans = 0;
        for (int i = 0; dp < nums.Length - 1; i++) {
            if (i > curr) {
                ans++;
                curr = dp;
            };
            dp = Math.Max(dp, nums[i] + i);
        };
        return ans;
    }
}

/*Greedy

The main idea is based on greedy.
Step 1: Let's say the range of the current jump is [left, right],
 dp is the dp position that all positions in [left, right] can reach.
Step 2: Once we reach to right, we trigger another jump with left = right + 1,
 right = dp, then repeat step 1 util we reach at the end.
 
Complexity
Time: O(N), where N <= 10^4 is the length of array nums
Space: O(1)*/

public class Solution {
    public int Jump(int[] nums) {
        int jumps = 0, dp = 0;
        int l = 0, r = 0;
        while (r < nums.Length - 1) {
            for (int i = l; i <= r; ++i)
                dp = Math.Max(dp, i + nums[i]);
            l = r + 1;
            r = dp;
            ++jumps;
        }
        return jumps;
    }
}

// Day 5
// 53. Maximum Subarray
// DP
// The idea is to maintain a running maximum smax and a current summation sum. 
// When we visit each num in nums, add num to sum, then update smax if necessary or reset sum to 0 if it becomes negative.
public class Solution {
    public int MaxSubArray(int[] nums) {
        if (nums.Length == 0) { return 0; }
        int sum = nums[0], dp = nums[0];
        for (int i=1; i<nums.Length; i++) {
            
            if ( sum >0 ) {
                sum += nums[i];
            }
            else{
                sum = nums[i];
            }
            
            dp = Math.Max(sum, dp);        
        }
        return dp;
    }
}

// Divide and Conquer
// The Divide-and-Conquer algorithm breaks nums into two halves and find the maximum subarray sum in them recursively.
// Well, the most tricky part is to handle the case that the maximum subarray spans the two halves.
// For this case, we use a linear algorithm: starting from the middle element and move to both ends 
// (left and right ends), record the maximum sum we have seen. 
// In this case, the maximum sum is finally equal to the middle element 
// plus the maximum sum of moving leftwards and the maximum sum of moving rightwards.
public class Solution {
    public int MaxSubArray(int[] nums) {
        return maxSubArray(nums, 0, nums.Length - 1);
    }
    private int maxSubArray(int[] nums, int l, int r) {
        if (l > r) {
            return Int32.MinValue;
        }
        int m = l + (r - l) / 2, ml = 0, mr = 0;
        int lmax = maxSubArray(nums, l, m - 1);
        int rmax = maxSubArray(nums, m + 1, r);
        for (int i = m - 1, sum = 0; i >= l; i--) {
            sum += nums[i];
            ml = Math.Max(sum, ml);
        }
        for (int i = m + 1, sum = 0; i <= r; i++) {
            sum += nums[i];
            mr = Math.Max(sum, mr);
        }
        return Math.Max(Math.Max(lmax, rmax), ml + mr + nums[m]);
    }
}

// 918. Maximum Sum Circular Subarray
/*
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
*/
public class Solution {
    public int MaxSubarraySumCircular(int[] nums) {
        int total = 0, maxSum = nums[0], curMax = 0, minSum = nums[0], curMin = 0;
        foreach (int a in nums) {
            curMax = Math.Max(curMax + a, a);
            maxSum = Math.Max(maxSum, curMax);
            curMin = Math.Min(curMin + a, a);
            minSum = Math.Min(minSum, curMin);
            total += a;
        }
        return maxSum > 0 ? Math.Max(maxSum, total - minSum) : maxSum;
    }
}

// Day 6
// 152. Maximum Product Subarray
public class Solution {
    public int MaxProduct(int[] nums) {
         int n = nums.Length, dp = nums[0], l = 0, r = 0;
        for (int i = 0; i < n; i++) {
            l =  (l == 0 ? 1 : l) * nums[i];
            r =  (r == 0 ? 1 : r) * nums[n - 1 - i];
            dp = Math.Max(dp, Math.Max(l, r));
        }
        return dp;
    }
}

// 1567. Maximum Length of Subarray With Positive Product
public class Solution {
    public int GetMaxLen(int[] nums) {
        int n = nums.Length;
        int[] pos = new int[n];
        int[] neg = new int[n];
        if (nums[0] > 0) pos[0] = 1;
        if (nums[0] < 0) neg[0] = 1;
        int ans = pos[0];
        for (int i = 1; i < n; i++) {
            if (nums[i] > 0) {
                pos[i] = 1 + pos[i - 1];
                neg[i] = neg[i - 1] > 0 ? 1 + neg[i - 1]:0;
            } else if (nums[i] < 0) {
                pos[i] = neg[i - 1] > 0 ? 1 + neg[i - 1]:0;
                neg[i] = 1 + pos[i - 1];
            }
            ans = Math.Max(ans, pos[i]);
        }
        return ans;
    }
}

// Day 7
// 1014. Best Sightseeing Pair
/*
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
*/
public class Solution {
    public int MaxScoreSightseeingPair(int[] values) {
        int res = 0, cur = 0;
        foreach (int a in values) {
            res = Math.Max(res, cur + a);
            cur = Math.Max(cur, a) - 1;
        }
        return res;
    }
}

// Soluton 2
public class Solution {
    public int MaxScoreSightseeingPair(int[] values) {
        int res = 0, imax = 0;
        for(int i = 0; i < values.Length; ++i) {
            res = Math.Max(res, imax + values[i] - i);
            imax = Math.Max(imax, values[i] + i);
        }
        return res;
    }
}

// 121. Best Time to Buy and Sell Stock
public class Solution {
    public int MaxProfit(int[] prices) {
        if (prices.Length <=1)
        {
            return 0;
        }
        int max_profit = 0, min_price = prices[0];
        for(int i=0;i<prices.Length;i++)
        {
            max_profit = Math.Max(max_profit,prices[i]-min_price);
            min_price = Math.Min(min_price,prices[i]);
        }
        return max_profit;
    }
}

// 122. Best Time to Buy and Sell Stock II
/* Solution 2: Greedy
Complexity:

Time: O(N)
Space: O(1)
*/
public class Solution {
    public int MaxProfit(int[] prices) {
        int n = prices.Length, maxProfit = 0; 
        for (int i=0; i<n-1 ; i++){
            if (prices[i+1] > prices[i])
                maxProfit += prices[i+1] - prices[i];
        }
            
        return maxProfit;
    }
}

// Day 8
// 309. Best Time to Buy and Sell Stock with Cooldown
public class Solution {
    public int MaxProfit(int[] prices) {
        int sold = 0;
        int rest = 0;
        int hold = Int32.MinValue;
        foreach (int price in prices) {
            int prev_sold = sold;
            sold = hold + price;
            hold = Math.Max(hold, rest - price);
            rest = Math.Max(rest, prev_sold);
        }
        return Math.Max(rest, sold);
    }
}

// 714. Best Time to Buy and Sell Stock with Transaction Fee
public class Solution {
    public int MaxProfit(int[] prices, int fee) {
        int cash = 0, hold = -prices[0];
        for (int i = 1; i < prices.Length; i++) {
            cash = Math.Max(cash, hold + prices[i] - fee);
            hold = Math.Max(hold, cash - prices[i]);
        }
        return cash;
    }
}

// Day 9
// 139. Word Break
// DP
// Time complexity O(n^2)
// Space complexity O(n^2)
public class Solution {
    public bool WordBreak(string s, IList<string> wordDict) {
        List<String> dict = new List<String>(wordDict);
        Dictionary<String, bool> mem = new Dictionary<String, bool>();
        return wordBreak(s, mem, dict);
    }
 
    private bool wordBreak(String s,
                              Dictionary<String, bool> mem, 
                             List<String> dict) {
        if (mem.ContainsKey(s)) return mem[s];
        if (dict.Contains(s)) {
            mem[s]= true;
            return true;
        }
        
        for (int i = 1; i < s.Length; ++i) {
            if (dict.Contains(s.Substring(i)) && wordBreak(s.Substring(0, i), mem, dict)) {
                mem[s]= true;
                return true;
            }
        }
        
        mem[s]= false;
        return false;
    }
}

// 42. Trapping Rain Water
// two-pointer
// Time complexity: O(n)
// Space complexity: O(1)
public class Solution {
    public int Trap(int[] height) {
    int n = height.Length;
    if (n == 0) return 0;
    int l = 0;
    int r = n - 1;
    int max_l = height[l];
    int max_r = height[r];
    int ans = 0;
    while (l < r) {      
      if (max_l < max_r) {
        ans += max_l - height[l];
        max_l = Math.Max(max_l, height[++l]);
      } else {
        ans += max_r - height[r];
        max_r = Math.Max(max_r, height[--r]);
      }
    }
    return ans;
    }
}
// DP
// l[i] := max(h[0:i+1])
// r[i] := max(h[i:n])
// ans = sum(min(l[i], r[i]) – h[i])
// Time complexity: O(n)
// Space complexity: O(n)
public class Solution {
    public int Trap(int[] height) {
    int n = height.Length;
    int[] l = new int[n];
    int[] r = new int[n];
    int ans = 0;
    for (int i = 0; i < n; ++i)
      l[i] = i == 0 ? height[i] : Math.Max(l[i - 1], height[i]);
    for (int i = n - 1; i >= 0; --i)
      r[i] = i == n - 1 ? height[i] : Math.Max(r[i + 1], height[i]);
    for (int i = 0; i < n; ++i)
      ans += Math.Min(l[i], r[i]) - height[i];
    return ans;
    }
}


// Day 10
// 413. Arithmetic Slices
public class Solution {
    public int NumberOfArithmeticSlices(int[] nums) {
        var slices = 0;
        for (int i = 2, prev = 0; i < nums.Length; i++)
            slices += (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) ? 
                    ++prev : 
                    (prev = 0);
        return slices;
    }
}

public class Solution {
    public int NumberOfArithmeticSlices(int[] nums) {
        var slices = 0;
        for (int i = 2, prev = 0; i < nums.Length; i++){
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) 
            {
                prev = prev + 1;
                slices = slices + prev;
            } 
            else {
                prev = 0;
            }
                      }
        return slices;
    }
}

public class Solution {
    public int NumberOfArithmeticSlices(int[] nums) {
        int curr = 0, sum = 0;
        for (int i=2; i<nums.Length; i++)
            if (nums[i]-nums[i-1] == nums[i-1]-nums[i-2]) {
                curr += 1;
                sum += curr;
            } else {
                curr = 0;
            }
        return sum;
    }
}
// 91. Decode Ways
public class Solution {
    public int NumDecodings(string s) {
        if (s == null || s.Length == 0) {
            return 0;
        }
        int n = s.Length;
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = s[0] != '0' ? 1 : 0;
        for (int i = 2; i <= n; i++) {
            int first = Int32.Parse(s.Substring(i - 1, 1));
            int second = Int32.Parse(s.Substring(i - 2, 2));
            if (first >= 1 && first <= 9) {
               dp[i] += dp[i-1];  
            }
            if (second >= 10 && second <= 26) {
                dp[i] += dp[i-2];
            }
        }
        return dp[n];
    }
}

public class Solution {
    public int NumDecodings(string s) {
        int n = s.Length, dp = 0, dp1 = 1, dp2 = 0;
        for (int i = n - 1; i >= 0; --i) {
            if (s[i] != '0') // Single digit
                dp += dp1;
            if (i+1 < s.Length && (s[i] == '1' || s[i] == '2' && s[i+1] <= '6')) // Two digits
                dp += dp2;
            dp2 = dp1;
            dp1 = dp;
            dp = 0;
        }
        return dp1;
    }
}

// Day 11
// 264. Ugly Number II
public class Solution {
    public int NthUglyNumber(int n) {
        if(n <= 0) return 0; // get rid of corner cases 
        if(n == 1) return 1; // base case
        int t2 = 0, t3 = 0, t5 = 0; //pointers for 2, 3, 5
        int[] k = new int[n];
        k[0] = 1;
        for(int i  = 1; i < n ; i ++)
        {
            k[i] = Math.Min(k[t2]*2,Math.Min(k[t3]*3,k[t5]*5));
            if(k[i] == k[t2]*2) t2++; 
            if(k[i] == k[t3]*3) t3++;
            if(k[i] == k[t5]*5) t5++;
        }
        return k[n-1];
    }
}

public class Solution {
    public int NthUglyNumber(int n) {
        int[] ugly = new int[n];
        ugly[0] = 1;
        int index2 = 0, index3 = 0, index5 = 0;
        int factor2 = 2, factor3 = 3, factor5 = 5;
        for(int i=1;i<n;i++){
            int min = Math.Min(Math.Min(factor2,factor3),factor5);
            ugly[i] = min;
            if(factor2 == min)
                factor2 = 2*ugly[++index2];
            if(factor3 == min)
                factor3 = 3*ugly[++index3];
            if(factor5 == min)
                factor5 = 5*ugly[++index5];
        }
        return ugly[n-1];
    }
}
// 96. Unique Binary Search Trees
public class Solution {
    public int NumTrees(int n) {
        /**
 * Taking 1~n as root respectively:
 *      1 as root: // of trees = F(0) * F(n-1)  // F(0) == 1
 *      2 as root: // of trees = F(1) * F(n-2) 
 *      3 as root: // of trees = F(2) * F(n-3)
 *      ...
 *      n-1 as root: // of trees = F(n-2) * F(1)
 *      n as root:   // of trees = F(n-1) * F(0)
 *
 * So, the formulation is:
 *      F(n) = F(0) * F(n-1) + F(1) * F(n-2) + F(2) * F(n-3) + ... + F(n-2) * F(1) + F(n-1) * F(0)
 */
        int[] dp = new int[n+1];
        dp[0] = dp[1] = 1;
        for (int i=2; i<=n; i++) {
            dp[i] = 0;
            for (int j=1; j<=i; j++) {
                dp[i] += dp[j-1] * dp[i-j];
            }
        }
        return dp[n];
    }
}


// Day 12
// 118. Pascal's Triangle
public class Solution {
    public IList<IList<int>> Generate(int numRows) {
        IList<IList<int>> result = new List<IList<int>>();        
        for( int i = 0; i < numRows; i++ )
        {
            IList<int> row = Enumerable.Repeat(1, i+1).ToList();
            for( int j = 1; j < i; j++ )
            {
                row[j] = result[i - 1][j] + result[i - 1][j - 1];
                
            }
            result.Add(row);
        }
        return result;
    }
}

// 119. Pascal's Triangle II
public class Solution {
    public IList<int> GetRow(int rowIndex) {
        int[] ans = new int[rowIndex+1];
        ans[0] = 1;
        ans[rowIndex] = 1;
        long temp = 1;
        int up = rowIndex;
        for(int i=1; i < rowIndex; i++){
            temp = temp * up / i;
            ans[i]=(int)temp;
            up--;
        }
        return ans;
    }
}

public class Solution {
    public IList<int> GetRow(int rowIndex) {
        int[] arr = new int[rowIndex + 1];
        Array.Fill(arr, 0);
        arr[0] = 1;
        
        for (int i = 1; i <= rowIndex; i++) 
            for (int j = i; j > 0; j--) 
                arr[j] = arr[j] + arr[j - 1];
        
        return arr.ToList();
    }
}
// Day 13
// 931. Minimum Falling Path Sum
/*
Solution: DP in place
Time complexity: O(mn)
Space complexity: O(mn)
*/
public class Solution {
    public int MinFallingPathSum(int[][] matrix) {
        int n = matrix.Length;
        int m = matrix[0].Length;    

        for (int i = 1; i < n; ++i)
          for (int j = 0; j < m; ++j) {
            int sum = matrix[i - 1][j];
            if (j > 0) sum = Math.Min(sum, matrix[i - 1][j - 1]);
            if (j < m - 1) sum = Math.Min(sum, matrix[i - 1][j + 1]);
            matrix[i][j] += sum;
          }

        return matrix[matrix.Length -1].Min();
    }
}
/*
Solution: DP
Time complexity: O(mn)
Space complexity: O(mn)
*/
public class Solution {
    public int MinFallingPathSum(int[][] matrix) {
        int n = matrix.Length;
    int m = matrix[0].Length;
    int[][] dp = new int[n+2][];
          
    for (int i = 0; i < n+2; ++i) {dp[i] = new int[m+2];}
    for (int i = 1; i <= n; ++i) {
      dp[i][0] = dp[i][m + 1] = Int32.MaxValue;
      for (int j = 1; j <= m; ++j)
        dp[i][j] = dp[i - 1][(0 + j - 1)..(0 + j + 2)].Min() + matrix[i - 1][j - 1];      
    }
    return dp[n][1..(dp.Length-1)].Min();
        
    
    }
}

// 120. Triangle
/*
Solution: DP
Time complexity: O(n^2)
Space complexity: O(1)*/
public class Solution {
    public int MinimumTotal(IList<IList<int>> triangle) {
        /*
        // [[2]                 [[2]
        //  [3, 4]]              [5, 6]
        //  [6, 5, 7]]           [11, 10, 11]
        //  [4, 1, 8, 3]]        [15, 11, 18, 14]]
        */
        int n = triangle.Count;  
 
        // t[i][j] := minTotalOf(i,j)
        // t[i][j] += min(t[i - 1][j], t[i - 1][j - 1])

        for (int i = 0; i < n; ++i)
          for (int j = 0; j <= i; ++j) {
            if (i == 0 && j == 0) continue; //row 0
            if (j == 0) triangle[i][j] += triangle[i - 1][j]; // row 1
            else if (j == i) triangle[i][j] += triangle[i - 1][j - 1]; 
            else triangle[i][j] += Math.Min(triangle[i - 1][j], triangle[i - 1][j - 1]);
          }

        return triangle[triangle.Count-1].Min();
    }
}

// Day 14
// 1314. Matrix Block Sum
/*
rangeSum[i + 1][j + 1] corresponds to cell (i, j);
rangeSum[0][j] and rangeSum[i][0] are all dummy values, 
which are used for the convenience of computation of DP state transmission formula.
Analysis:
Time & space: O(m * n).
*/
public class Solution {
    public int[][] MatrixBlockSum(int[][] mat, int k) {
        int m = mat.Length, n = mat[0].Length;
        int[][] rangeSum = new int[m + 1][];
        for (int i = 0; i < m+1; ++i){ rangeSum[i] = new int[n + 1];}
        for (int i = 0; i < m; ++i){
            for (int j = 0; j < n; ++j){
                rangeSum[i + 1][j + 1] = rangeSum[i + 1][j] + rangeSum[i][j + 1] - rangeSum[i][j] + mat[i][j];
            }
        }    
        int[][] ans = new int[m][];
        for (int i = 0; i < m; ++i){
            ans[i] = new int[n];
             for (int j = 0; j < n; ++j) {
                int r1 = Math.Max(0, i - k), c1 = Math.Max(0, j - k), r2 = Math.Min(m, i + k + 1), c2 = Math.Min(n, j + k + 1);
                ans[i][j] = rangeSum[r2][c2] - rangeSum[r2][c1] - rangeSum[r1][c2] + rangeSum[r1][c1];
            }
        }
           
        return ans;
    }
}

// 304. Range Sum Query 2D - Immutable
/*
Complexity
Constructor: Time & Space: O(m*n), where m is the number of rows, n is the number of columns in the grid
sumRegion: Time & Space: O(1)
*/
public class NumMatrix {

    private int[][] dp;

    public NumMatrix(int[][] matrix) {
        if (matrix.Length == 0 || matrix[0].Length == 0) return;
        dp = new int[matrix.Length + 1][];
        for (int r = 0; r < matrix.Length+1; r++) {
        dp[r] = new int[matrix[0].Length + 1];}
    for (int r = 0; r < matrix.Length; r++) {
        for (int c = 0; c < matrix[0].Length; c++) {
            dp[r + 1][c + 1] = dp[r + 1][c] + dp[r][c + 1] + matrix[r][c] - dp[r][c];
        }
    }
    }
    
    public int SumRegion(int row1, int col1, int row2, int col2) {
        return dp[row2 + 1][col2 + 1] - dp[row1][col2 + 1] - dp[row2 + 1][col1] + dp[row1][col1];
    }
}

/**
 * Your NumMatrix object will be instantiated and called as such:
 * NumMatrix obj = new NumMatrix(matrix);
 * int param_1 = obj.SumRegion(row1,col1,row2,col2);
 */

// Day 15
// 62. Unique Paths
public class Solution {  
    public int UniquePaths(int m, int n) {
        int[][] grid = new int[n+1][];
        for(int i = 0; i<= n; i++){ grid[i] = new int[m+1]; Array.Fill(grid[i], 0);}
        for(int i = 1; i<= n; i++){
            for(int j = 1; j<= m; j++){
                if(i==1 && j==1){
                    grid[1][1] = 1;
                    continue;
                } else {
                    grid[i][j] = grid[i][j-1] + grid[i-1][j];
                }
            }
        }
        return grid[n][m];

    } 
}

public class Solution {  
    public int UniquePaths(int m, int n) {
        int[][] grid = new int[m][];
        for(int i = 0; i<m; i++){
            grid[i] = new int[n];
            for(int j = 0; j<n; j++){
                if(i==0||j==0)
                    grid[i][j] = 1;
                else
                    grid[i][j] = grid[i][j-1] + grid[i-1][j];
            }
        }
        return grid[m-1][n-1];
    } 
}
// DFS
public class Solution {
    private int[][] dp;
    private int m;
    private int n;
  
    public int UniquePaths(int m, int n) {
        this.dp = new int[m][];    
        this.m = m;
        this.n = n;
        for(int i=0;i<m;i++){dp[i]=new int[n];}
        return dfs(0, 0);
    }
    
    private int dfs(int x, int y) {
        if (x > m - 1 || y > n - 1)
          return 0;
        if (x == m - 1 && y == n - 1)
          return 1;
        if (dp[x][y] == 0)     
          dp[x][y] = dfs(x + 1, y) + dfs(x , y + 1);
        return dp[x][y];
    } 
}
// 63. Unique Paths II
// DP In-place
/*
Complexity Analysis
Time Complexity: O(M times N)O(M×N). 
The rectangular grid given to us is of size M times N and we process each cell just once.
Space Complexity: O(1). 
We are utilizing the obstacleGrid as the DP array. Hence, no extra space.
*/
public class Solution {
    public int UniquePathsWithObstacles(int[][] obstacleGrid) {
         int R = obstacleGrid.Length;
        int C = obstacleGrid[0].Length;

        // If the starting cell has an obstacle, then simply return as there would be
        // no paths to the destination.
        if (obstacleGrid[0][0] == 1) {
            return 0;
        }

        // Number of ways of reaching the starting cell = 1.
        obstacleGrid[0][0] = 1;

        // Filling the values for the first column
        for (int i = 1; i < R; i++) {
            obstacleGrid[i][0] = (obstacleGrid[i][0] == 0 && obstacleGrid[i - 1][0] == 1) ? 1 : 0;
        }

        // Filling the values for the first row
        for (int i = 1; i < C; i++) {
            obstacleGrid[0][i] = (obstacleGrid[0][i] == 0 && obstacleGrid[0][i - 1] == 1) ? 1 : 0;
        }

        // Starting from cell(1,1) fill up the values
        // No. of ways of reaching cell[i][j] = cell[i - 1][j] + cell[i][j - 1]
        // i.e. From above and left.
        for (int i = 1; i < R; i++) {
            for (int j = 1; j < C; j++) {
                if (obstacleGrid[i][j] == 0) {
                    obstacleGrid[i][j] = obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1];
                } else {
                    obstacleGrid[i][j] = 0;
                }
            }
        }
        // Return value stored in rightmost bottommost cell. That is the destination.
        return obstacleGrid[R - 1][C - 1];
    }
}

// DP
public class Solution {
    public int UniquePathsWithObstacles(int[][] obstacleGrid) {
        int n = obstacleGrid.Length;
        if (n == 0) return 0;
        int m = obstacleGrid[0].Length;
        
        // f[i][j] = paths(i, j)
        // INT_MIN -> not solved yet, solution unknown
        f_ = new int[n + 1][]; 
        for(int i=0; i< n+1; i++){f_[i] = new int[m + 1]; Array.Fill(f_[i],Int32.MinValue);}
        return paths(m, n, obstacleGrid);
    }
private int[][] f_;
    
    int paths(int x, int y, int[][] o) {
        // Out of bound, return 0.
        if (x <= 0 || y <= 0) return 0;
        
        // Reaching the starting point.
        // Note, there might be an obstacle here as well.
        if (x == 1 && y == 1) return 1 - o[0][0];
        
        // Already solved, return the answer.
        if (f_[y][x] != Int32.MinValue) return f_[y][x];
        
        // There is an obstacle on current block, no path
        if (o[y - 1][x - 1] == 1) {
            f_[y][x] = 0;
        } else {
            // Recursively find paths.
            f_[y][x] = paths(x - 1, y, o) + paths(x, y - 1, o);
        }
        
        // Return the memorized answer.
        return f_[y][x];
    }
}
// Day 16
// 64. Minimum Path Sum
/*
C++ / DP
Time complexity: O(mn)
Space complexity: O(1)
*/
public class Solution {
    public int MinPathSum(int[][] grid) {
        int m = grid.Length;
        if (m == 0) return 0;
        int n = grid[0].Length;
        
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == 0 && j == 0) continue;
                if (i == 0) 
                    grid[i][j] += grid[i][j - 1];
                else if (j == 0)
                    grid[i][j] += grid[i - 1][j];
                else
                    grid[i][j] += Math.Min(grid[i][j - 1], grid[i - 1][j]);
            }
        
        return grid[m - 1][n - 1];
    }
}
/*
C++ / Recursion with memoization
Time complexity: O(mn)
Space complexity: O(mn)
*/
public class Solution {
    public int MinPathSum(int[][] grid) {
       int m = grid.Length;
        if (m == 0) return 0;
        int n = grid[0].Length;
        
        s_ = new int[m][];
        for(int i = 0; i<m;i++){s_[i] = new int[n]; Array.Fill(s_[i], 0);}
        
        return minPathSum(grid, n - 1, m - 1, n, m);
    }    
private int minPathSum(int[][] grid, 
                   int x, int y, int n, int m) {        
        if (x == 0 && y == 0) return grid[y][x];
        if (x < 0 || y < 0) return Int32.MaxValue;
        if (s_[y][x] > 0) return s_[y][x];
        
        int ans = grid[y][x] + Math.Min(minPathSum(grid, x - 1, y, n, m),
                                   minPathSum(grid, x, y - 1, n, m));
        return s_[y][x] = ans;
    }
    
    int[][] s_;
}
// 221. Maximal Square
// Better DP
/*
Complexity Analysis
Time complexity : O(mn). Single pass.
Space complexity : O(n). Another array which stores elements in a row is used for dp.*/
public class Solution {
    public int MaximalSquare(char[][] matrix) {
        int rows = matrix.Length, cols = rows > 0 ? matrix[0].Length : 0;
        int[] dp = new int[cols + 1];
        int maxsqlen = 0, prev = 0;
        for (int i = 1; i <= rows; i++) {
            for (int j = 1; j <= cols; j++) {
                int temp = dp[j];
                if (matrix[i - 1][j - 1] == '1') {
                    dp[j] = Math.Min(Math.Min(dp[j - 1], prev), dp[j]) + 1;
                    maxsqlen = Math.Max(maxsqlen, dp[j]);
                } else {
                    dp[j] = 0;
                }
                prev = temp;
            }
        }
        return maxsqlen * maxsqlen;
    }
}

// DP
/*Complexity Analysis
Time complexity : O(mn). Single pass.
Space complexity : O(mn). Another matrix of same size is used for dp.
*/
public class Solution {
    public int MaximalSquare(char[][] matrix) {
        int rows = matrix.Length, cols = rows > 0 ? matrix[0].Length : 0;
        int[][] dp = new int[rows + 1][];
        for (int i = 0; i <= rows; i++) {dp[i] = new int[cols+1];}
        int maxsqlen = 0;
        for (int i = 1; i <= rows; i++) {
            for (int j = 1; j <= cols; j++) {
                if (matrix[i-1][j-1] == '1'){
                    dp[i][j] = Math.Min(Math.Min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1;
                    maxsqlen = Math.Max(maxsqlen, dp[i][j]);
                }
            }
        }
        return maxsqlen * maxsqlen;
    }
}

// Day 17
// 5. Longest Palindromic Substring
/*Approach : Expand Around Center

Complexity Analysis
Time complexity : O(n^2). Since expanding a palindrome around its center could take O(n) time, 
Space complexity : O(1).*/
public class Solution {
    public string LongestPalindrome(string s) {
       if (s == null || s.Length < 1) return "";
    int start = 0, end = 0;
    for (int i = 0; i < s.Length; i++) {
        int len1 = expandAroundCenter(s, i, i);
        int len2 = expandAroundCenter(s, i, i + 1);
        int len = Math.Max(len1, len2);
        if (len > end - start) {
            start = i - (len - 1) / 2;
            end = i + len / 2;
        }
    }
    return s.Substring(start, end + 1 - start);
}

private int expandAroundCenter(String s, int left, int right) {
    int L = left, R = right;
    while (L >= 0 && R < s.Length && s[L] == s[R]) {
        L--;
        R++;
    }
    return R - L - 1;
}
}

// 516. Longest Palindromic Subsequence
// Complexity
// Time: O(N^2), where N <= 1000 is length of string s.
// Space: O(N)
public class Solution {
    public int LongestPalindromeSubseq(string s) {
            int len = s.Length;
            int[] dp = new int[len];
            int pre=0;
          for(int i=len-1;i>=0;i--)
          {   
              for(int j=i;j<len;j++)
              {  int temp = dp[j];
                 if(i==j){dp[j]=1;pre=temp;continue;}
                   if(s[i] == s[j]) {
                        dp[j] = pre+ 2;
                    }
                    else {
                        dp[j] = Math.Max(dp[j], dp[j-1]);
                    }
                 pre=temp;
              }
          }
            return dp[len-1];
    }
}
/*
dp[i][j]: the longest palindromic subsequence's length of substring(i, j), 
here i, j represent left, right indexes in the string
State transition:
dp[i][j] = dp[i+1][j-1] + 2 if s.charAt(i) == s.charAt(j)
otherwise, dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1])
Initialization: dp[i][i] = 1
*/
public class Solution {
    public int LongestPalindromeSubseq(string s) {
           int[][] dp = new int[s.Length][];
        for (int i =0 ; i < s.Length; i++) {
            dp[i] = new int[s.Length];}
        for (int i = s.Length - 1; i >= 0; i--) {
            dp[i][i] = 1;
            for (int j = i+1; j < s.Length; j++) {
                if (s[i] == s[j]) {
                    dp[i][j] = dp[i+1][j-1] + 2;
                } else {
                    dp[i][j] = Math.Max(dp[i+1][j], dp[i][j-1]);
                }
            }
        }
        return dp[0][s.Length-1];
    }
}

// Day 18
// 300. Longest Increasing Subsequence
/*
Solution 2: DP + Binary Search / Patience Sort

dp[i] := smallest tailing number of a increasing subsequence of length i + 1.
dp is an increasing array, we can use binary search to find the index to insert/update the array.
ans = len(dp)

Time complexity: O(nlogn)
Space complexity: O(n)
*/
public class Solution {
    public int LengthOfLIS(int[] nums) {       
    int n = nums.Length;
    if (n == 0) return 0;    
    List<int> dp = new List<int>();
    for (int i = 0; i < n; ++i) {
      int it = binarySearch(dp, nums[i]);
        if (it < 0) it = ~it;
      if (it == dp.Count)
        dp.Add(nums[i]);
      else
        dp[it] = nums[i];
    }
    return dp.Count;
    
    }
    
    int binarySearch(List<int> nums, int x) {
        int l = 0, r = nums.Count;
        while (l < r) {
            int m = l+ (r-l) / 2;
            if (nums[m] >= x)
                r = m;
            else
                l = m + 1;
        }
        return l;
    }
}

// Use C# Binary Seach + Patience Sorting
// This algorithm is actually Patience sorting.
// It might be easier for you to understand how it works if you think about it as piles of cards instead of tails.
// The number of piles is the length of the longest subsequence.
// For more info see Princeton lecture.
public class Solution {
    public int LengthOfLIS(int[] nums) {
        List<int> dp = new List<int>();
        
        foreach (int x in nums) {
            int it = dp.BinarySearch(x); // directly use C# binary search here
            if (it < 0) { it = ~it;}
            if (it == dp.Count){
                dp.Add(x);   
            }
            else{
                dp[it] = x;
            }
        }
        return dp.Count;
    }
}
// Binary Search / Patience sorting
public class Solution {
    public int LengthOfLIS(int[] nums) {
        int[] tails = new int[nums.Length];
        int size = 0;
        foreach (int x in nums) {
            int i = 0, j = size;
            while (i != j) { // use binary search here!
                int m = (i + j) / 2;
                if (tails[m] < x)
                    i = m + 1;
                else
                    j = m;
            }
            tails[i] = x;
            if (i == size) ++size;
        }
        return size;
    }
}
// Time Complexity: O(n^2) sort of slow
public class Solution {
    public int LengthOfLIS(int[] nums) {
        if (nums.Length == 0) return 0;
        int n = nums.Length;
        int[] f = new int[n]; Array.Fill(f, 1);
        for (int i = 1; i < n; ++i)
            for (int j = 0; j < i; ++j)
                if (nums[i] > nums[j])
                    f[i] = Math.Max(f[i], f[j] + 1);
        return f.Max();
       
    }
}

// 376. Wiggle Subsequence
// Approach # Space-Optimized Dynamic Programming
// Complexity Analysis
// Time complexity : O(n). Only one pass over the array length.
// Space complexity : O(1). Constant space is used.
public class Solution {
    public int WiggleMaxLength(int[] nums) {
         if (nums.Length < 2)
            return nums.Length;
        int down = 1, up = 1;
        for (int i = 1; i < nums.Length; i++) {
            if (nums[i] > nums[i - 1])
                up = down + 1;
            else if (nums[i] < nums[i - 1])
                down = up + 1;
        }
        return Math.Max(down, up);
    }
}
// Approach # Greedy Approach
// Complexity Analysis
// Time complexity : O(n). We traverse the given array once.
// Space complexity : O(1). No extra space is used.
public class Solution {
    public int WiggleMaxLength(int[] nums) {
        if (nums.Length < 2)
            return nums.Length;
        int prevdiff = nums[1] - nums[0];
        int count = prevdiff != 0 ? 2 : 1;
        for (int i = 2; i < nums.Length; i++) {
            int diff = nums[i] - nums[i - 1];
            if ((diff > 0 && prevdiff <= 0) || (diff < 0 && prevdiff >= 0)) {
                count++;
                prevdiff = diff;
            }
        }
        return count;
    }
}

// Day 19
// 392. Is Subsequence
// IndexOf AND Substring is faster
public class Solution {
    public bool IsSubsequence(string s, string t) {
       if(t.Length < s.Length) return false;
        
        foreach (char ch in s.ToCharArray()) {
            int index = Array.IndexOf(t.ToCharArray(), ch);
            if(index < 0){
               return false; 
            }
            t = t.Substring(index+1);
        }
        return true;
    }
}

public class Solution {
    public bool IsSubsequence(string s, string t) {
        if (s.Length == 0) return true;
        int indexS = 0, indexT = 0;
        while (indexT < t.Length) {
            if (t[indexT] == s[indexS]) {
                indexS++;
                if (indexS == s.Length) return true;
            }
            indexT++;
        }
        return false;
    }
}
/**
 * Follow-up
 * If we check each sk in this way, then it would be O(kn) time where k is the number of s and t is the length of t. 
 * This is inefficient. 
 * Since there is a lot of s, it would be reasonable to preprocess t to generate something that is easy to search for if a character of s is in t. 
 * Sounds like a HashMap, which is super suitable for search for existing stuff. 
 */
public class Solution {
    public bool IsSubsequence(string s, string t) {
        if (s == null || t == null) return false;
    
    Dictionary<char, List<int>> map = new Dictionary<char, List<int>>(); //<character, index>
    
    //preprocess t
    for (int i = 0; i < t.Length; i++) {
        char curr = t[i];
        if (!map.ContainsKey(curr)) {
            map[curr] = new List<int>();
        }
        map[curr].Add(i);
    }
    
    int prev = -1;  //index of previous character
    for (int i = 0; i < s.Length; i++) {
        char c = s[i];
        
        if (!map.ContainsKey(c) || map[c] == null)  {
            return false;
        } else {
            List<int> list = map[c];
            prev = binarySearch(prev, list, 0, list.Count - 1);
            if (prev == -1) {
                return false;
            }
            prev++;
        }
    }
    
    return true;
}

private int binarySearch(int index, List<int> list, int start, int end) {
    while (start <= end) {
        int mid = start + (end - start) / 2;
        if (list[mid] < index) {
            start = mid + 1;
        } else {
            end = mid - 1;
        }
    }
    
    return start == list.Count ? -1 : list[start];
}
}

// 1143. Longest Common Subsequence
/*
Solution: DP

Use dp[i][j] to represent the length of longest common sub-sequence of text1[0:i] and text2[0:j]
dp[i][j] = dp[i – 1][j – 1] + 1 if text1[i – 1] == text2[j – 1] else max(dp[i][j – 1], dp[i – 1][j])

Time complexity: O(mn)
Space complexity: O(mn) -> O(n)
*/

public class Solution {
    public int LongestCommonSubsequence(string text1, string text2) {
        int m = text1.Length;
        int n = text2.Length;    
        int[] dp1 = new int[n+1] , dp2 = new int[n+1];    
        for (int i = 0; i < m; ++i) {
         
          for (int j = 0; j < n; ++j) {        
            
            if (text1[i] == text2[j])
              // dp[i + 1][j + 1] = dp[i][j] + 1
              dp2[j + 1] = dp1[j] + 1; 
            else
              // dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
              dp2[j + 1] = Math.Max(dp1[j + 1], dp2[j]);
            
          }    
            
            int[] temp = dp1;
            dp1 = dp2;
            dp2 = temp; 
        }    
        return dp1[n]; // dp[m][n]

    
    }
}

public class Solution {
    public int LongestCommonSubsequence(string text1, string text2) {
        int m = text1.Length;
        int n = text2.Length;    
        int[] dp = new int[n+1];    
        for (int i = 0; i < m; ++i) {
          int prev = 0; // dp[i][j]
          for (int j = 0; j < n; ++j) {        
            int curr = dp[j + 1]; // dp[i][j + 1]
            if (text1[i] == text2[j])
              // dp[i + 1][j + 1] = dp[i][j] + 1
              dp[j + 1] = prev + 1; 
            else
              // dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
              dp[j + 1] = Math.Max(curr, dp[j]);
            prev = curr;
          }    
        }    
        return dp[n]; // dp[m][n]
    }
}

// 72. Edit Distance
// Iterative Dynamic Programming
public class Solution {
    public int MinDistance(string word1, string word2) {
        int l1 = word1.Length;
        int l2 = word2.Length;
        // d[i][j] := minDistance(word1[0:i - 1], word2[0:j - 1]);
        int[][] d = new int[l1+1][];
        
        for (int i = 0; i <= l1; ++i){
             d[i] = new int[l2+1];
            d[i][0] = i;
        }
           
        for (int j = 0; j <= l2; ++j){
            d[0][j] = j;
        }
            
        for (int i = 1; i <= l1; ++i){
            for (int j = 1; j <= l2; ++j) {
                int c = (word1[i - 1] == word2[j - 1]) ? 0 : 1;
                d[i][j] = Math.Min(d[i - 1][j - 1] + c, 
                              Math.Min(d[i][j - 1], d[i - 1][j]) + 1);
            }
        }
                   
        return d[l1][l2];
    }
}
// Recursive Dynamic Programming
public class Solution {
    public int MinDistance(string word1, string word2) {
        int l1 = word1.Length;
        int l2 = word2.Length;
        d_ = new int[l1 + 1][];
        for(int i=0; i<l1+1; i++){ d_[i] = new int[l2 + 1] ; Array.Fill(d_[i],-1);}
        return minDistance(word1, word2, l1, l2);
    }
    
    private int[][] d_;
    // minDistance from word1[0:l1-1] to word2[0:l2-1]
    public int minDistance(string word1, string word2, int l1, int l2) {
        if (l1 == 0) return l2;
        if (l2 == 0) return l1;
        if (d_[l1][l2] >= 0) return d_[l1][l2];
        
        int ans;
        if (word1[l1 - 1] == word2[l2 - 1])
            ans = minDistance(word1, word2, l1 - 1, l2 - 1);
        else 
            ans = Math.Min(minDistance(word1, word2, l1 - 1, l2 - 1),
                  Math.Min(minDistance(word1, word2, l1 - 1, l2), 
                      minDistance(word1, word2, l1, l2 - 1))) + 1;
        
        return d_[l1][l2] = ans;        
    }
}

// Day 20
// 322. Coin Change
/* Knapsack problem
This is a classic knapsack problem. Honestly, I'm not good at knapsack problem, it's really tough for me.

dp[i][j] : the number of combinations to make up amount j by using the first i types of coins
State transition:

not using the ith coin, only using the first i-1 coins to make up amount j, then we have dp[i-1][j] ways.
using the ith coin, since we can use unlimited same coin, we need to know how many ways to make up amount j - coins[i-1] by using first i coins(including ith), which is dp[i][j-coins[i-1]]
Initialization: dp[i][0] = 1

Once you figure out all these, it's easy to write out the code:
*/
public class Solution {
    public int Change(int amount, int[] coins) {
         int[][] dp = new int[coins.Length+1][];
       
        for (int i = 0; i < coins.Length+1; i++) {
            dp[i] = new int[amount+1];
        }
         dp[0][0] = 1;
        for (int i = 1; i <= coins.Length; i++) {
            dp[i][0] = 1;
            for (int j = 1; j <= amount; j++) {
                dp[i][j] = dp[i-1][j] + (j >= coins[i-1] ? dp[i][j-coins[i-1]] : 0);
            }
        }
        return dp[coins.Length][amount];      
    }
}
// Now we can see that dp[i][j] only rely on dp[i-1][j] and dp[i][j-coins[i]], then we can optimize the space by only using one-dimension array.
public class Solution {
    public int Change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        foreach (int coin in coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] += dp[i-coin];
            }
        }
        return dp[amount];
     
    }
}
// 518. Coin Change 2
/*Solution : DP
Time complexity: O(n*amount)
Space complexity: O(amount)
*/
public class Solution {
    public int CoinChange(int[] coins, int amount) {
        int[] dp = Enumerable.Repeat(Int32.MaxValue, amount+1).ToArray();
        
        dp[0] = 0;
        foreach (int coin in coins) {
            for (int i = coin; i <= amount; ++i)
                if (dp[i - coin] != Int32.MaxValue)  
                    dp[i] = Math.Min(dp[i], dp[i - coin] + 1);
        }
        
        return dp[amount] == Int32.MaxValue ? -1 : dp[amount];
    }
}
/*
DFS unfortunely is TLE
Time complexity: O(amount^n/(coin_0*coin_1*…*coin_n))
Space complexity: O(n)
*/
public class Solution {
    int[] c;
    public int CoinChange(int[] coins, int amount) {
        // Sort coins in desending order        
        //Array.Sort(coins);// Sort array in ascending order.
        //Array.Reverse(coins); // reverse array
        // Sort the array in decreasing order and return a array
        c = coins.OrderByDescending(c => c).ToArray();
        //ans = Int32.MaxValue;
        coinChange(c, 0, amount, 0);
        return ans == Int32.MaxValue ? -1 : ans;
    }
int ans = Int32.MaxValue;
public void coinChange(int[] coins, int s, int amount, int count) {
        if (amount == 0) {
            ans = Math.Min(ans, count);
            return;
        }
        
        if (s == coins.Length) return;
        
       int coin = coins[s];

        for (int k = amount / coin; k >= 0 && count + k < ans; k--){
            coinChange(coins, s+1, amount - k*coin, count + k);
        }
            
    }  
//This is actually works! and relatively fast.
/*#Recursive Method:#
The idea is very classic dynamic programming: think of the last step we take. 
Suppose we have already found out the best way to sum up to amount a, 
then for the last step, we can choose any coin type which gives us a remainder r where r = a-coins[i] 
for all i's. For every remainder, go through exactly the same process as before until 
either the remainder is 0 or less than 0 (meaning not a valid solution). 
With this idea, the only remaining detail is to store the minimum number of coins 
needed to sum up to r so that we don't need to recompute it over and over again.*/
public class Solution {
    public int CoinChange(int[] coins, int amount) {
        if(amount<1) return 0;
    return helper(coins, amount, new int[amount]);
}

private int helper(int[] coins, int rem, int[] count) { // rem: remaining coins after the last step; count[rem]: minimum number of coins to sum up to rem
    if(rem<0) return -1; // not valid
    if(rem==0) return 0; // completed
    if(count[rem-1] != 0) return count[rem-1]; // already computed, so reuse
    int min = Int32.MaxValue;
    foreach(int coin in coins) {
        int res = helper(coins, rem-coin, count);
        if(res>=0 && res < min)
            min = 1+res;
    }
    count[rem-1] = (min==Int32.MaxValue) ? -1 : min;
    return count[rem-1];
}
}

// Day 21
// 377. Combination Sum IV
// Recursion + Memorization
public class Solution {
    public int CombinationSum4(int[] nums, int target) {
        m_ = new int[target + 1]; Array.Fill(m_, -1);
        m_[0] = 1;
        return dp(nums, target);
    }    
    private int dp(int[] nums, int target) {
            if (target < 0) return 0;
            if (m_[target] != -1) return m_[target];
            int ans = 0;
            foreach (int num in nums)
                ans += dp(nums, target - num);
            return m_[target] = ans;
        }
    int[] m_;
}
// DP
public class Solution {
    public int CombinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1]; Array.Fill(dp, 0); // dp[i] # of combinations sum up to i
        dp[0] = 1;
        for (int i = 1; i <= target; ++i)
            foreach (int num in nums)
                if (i - num >= 0)
                    dp[i] += dp[i - num];           
        return dp[target];
    }
}
// 343. Integer Break
public class Solution {
    public int IntegerBreak(int n) {
        //dp[i] means output when input = i, e.g. dp[4] = 4 (2*2),dp[8] = 18 (2*2*3)...
        if(n > 3) n++;
        int[] dp = new int[n+1];
        dp[1] = 1;
        // fill the entire dp array
        for(int i = 2; i <=n; i++) {
             //let's say i = 8, we are trying to fill dp[8]:
             //if 8 can only be broken into 2 parts, the answer could be among 1 * 7, 2 * 6, 3 * 5, 4 * 4... 
             //but these numbers can be further broken. 
             //so we have to compare 1 with dp[1], 7 with dp[7], 2 with dp[2], 6 with dp[6]...etc
            for(int j = 1; j < i; j++) {
                // use Math.max(dp[i],....)  so dp[i] maintain the greatest value
                dp[i] = Math.Max(dp[i], j * dp[i-j]);
            }
        }
        return dp[n];
    }
}

// 279. Perfect Squares
/*Solution : DP
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
*/
public class Solution {
    public int NumSquares(int n) {
        int[] dp = new int[n + 1]; Array.Fill(dp, (Int32.MaxValue >> 1));
        dp[0] = 0;
        for (int i = 1; i <= n; ++i)
        for (int j = 1; j * j <= i; ++j) 
            dp[i] = Math.Min(dp[i], dp[i - j * j] + 1);
        return dp[n];
    }
}
