// 2420. Find All Good Indices
/*Solution: Prefix Sum
Let before[i] = length of longest non-increasing subarray ends of nums[i].
Let after[i] = length of longest non-decreasing subarray ends of nums[i].

An index is good if nums[i – 1] >= k and nums[i + k] >= k

Time complexity: O(n + (n – 2*k))
Space complexity: O(n)*/
public class Solution {
    public IList<int> GoodIndices(int[] nums, int k) {
        int n = nums.Length;
    int[] before = new int[n];Array.Fill(before, 1);
     int[] after = new int[n];Array.Fill(after, 1);
    for (int i = 1; i < n; ++i) {
      if (nums[i] <= nums[i - 1])
        before[i] = before[i - 1] + 1;      
      if (nums[i] >= nums[i - 1])
        after[i] = after[i - 1] + 1;    
    }
    IList<int> ans = new List<int>();
    for (int i = k; i + k < n; ++i) {
      if (before[i - 1] >= k && after[i + k] >= k)
        ans.Add(i);
    }
    return ans;
    }
}
// 2419. Longest Subarray With Maximum Bitwise AND
/*Solution: Find the largest number
a & b <= a
a & b <= b
if b > a, a & b < b, we choose to start a new sequence of “b” instead of continuing with “ab”

Basically, we find the largest number in the array and count the longest sequence of it. Note, there will be some tricky cases like.
b b b b a b
b a b b b b
We need to return 4 instead of 1.

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int LongestSubarray(int[] nums) {
        int ans = 0;
    int best = 0;
    for (int i = 0, l = 0; i < nums.Length; ++i) {
      if (nums[i] > best) {
        best = nums[i]; 
        ans = l = 1;
      } else if (nums[i] == best) {
        ans = Math.Max(ans, ++l);
      } else {
        l = 0;
      }
    }    
    return ans;
    }
}
// 2418. Sort the People
/*Solution: Zip and sort

Time complexity: O(nlogn)
Space complexity: O(n)*/
public class Solution {
    public string[] SortPeople(string[] names, int[] heights) {
        List<KeyValuePair<int, string>> A = new List<KeyValuePair<int, string>>();
        int N = names.Length;
        for(int i = 0; i < N; i++) {
            A.Add(new KeyValuePair<int, string>(heights[i], names[i]));
        }

        A = A.OrderByDescending(x => x.Key).ToList();

        string[] ans = new string[N];
        for(int i = 0; i < N; i++) {
            ans[i] = A[i].Value;
        }
        return ans;
    }
}

public class Solution {
    public string[] SortPeople(string[] names, int[] heights) {
        List<KeyValuePair<int, string>> A = new List<KeyValuePair<int, string>>();
        int N = names.Length;
        for(int i = 0; i < N; i++) {
            A.Add(new KeyValuePair<int, string>(heights[i], names[i]));
        }

        A = A.OrderByDescending(x => x.Key).ToList();

        List<string> ans = new List<string>();
        for(int i = 0; i < N; i++) {
            ans.Add(A[i].Value);
        }
        return ans.ToArray();
    }
}
// HashMap
public class Solution {
    public string[] SortPeople(string[] names, int[] heights) {
        Dictionary<int, String> map = new Dictionary<int, String>();
        for (int i = 0; i < names.Length; i++) {
            map[heights[i]]= names[i];
        }        
        Array.Sort(heights);
        String[] result = new String[heights.Length];
        int index = 0;
        for (int i = heights.Length - 1; i >= 0; i--) {
            result[index] = map[heights[i]];
            index++;
        }
        return result;
    }
}
// 2416. Sum of Prefix Scores of Strings
/*Solution: Trie
Insert all the words into a tire 
whose node val is the number of substrings that have the current prefix.

During query time, sum up the values along the prefix path.

Time complexity: O(sum(len(word))
Space complexity: O(sum(len(word))*/
class Trie {
  Dictionary<int, Trie> ch = new Dictionary<int, Trie>();
  int cnt = 0;
 public void insert(string s) {
   Trie cur = this;
    foreach (char c in s) {
      if (!cur.ch.ContainsKey(c - 'a')) 
        cur.ch[c - 'a'] = new Trie();
      cur = cur.ch[c - 'a'];
      ++cur.cnt;
      }
  }
  public int query(string s) {
    Trie cur = this;
    int ans = 0;
     foreach (char c in s) {
      cur = cur.ch[c - 'a'];
      ans += cur.cnt;
    }
    return ans;
  }
};
public class Solution {
    public int[] SumPrefixScores(string[] words) {
        Trie root = new Trie();
        foreach (String word in words) {
            root.insert(word);
        }
       // int n = words.Length, i = 0;
        List<int> ans = new List<int>();
        foreach (String word in words) {
            ans.Add(root.query(word));
        }
        return ans.ToArray();
    }
}
/*Build Trie and accumulate the frequencies of each pefix at the same time; 
then search each word and compute the corresponding score.
Analysis:

Time & space: O(n * w), where n = words.length,
 w = average size of word in `words``.*/
class Trie {
  Dictionary<char, Trie> kids = new Dictionary<char, Trie>();
  int cnt = 0;
 public void insert(string s) {
   Trie t = this;
        foreach (char c in s) {
            if (!t.kids.ContainsKey(c)) {
                t.kids[c] = new Trie();
            }
            t = t.kids[c];
            t.cnt += 1;
        }
      
  }
  public int query(string s) {
    Trie t = this;
        int score = 0;
        foreach (char c in s) {
            if (t.kids[c] == null) {
                 return score;
            }
            t = t.kids[c];
            score += t.cnt;
        }
        return score;
     
  }
};
public class Solution {
    public int[] SumPrefixScores(string[] words) {
        Trie root = new Trie();
        foreach (String word in words) {
            root.insert(word);
        }
       // int n = words.Length, i = 0;
        List<int> ans = new List<int>();
        foreach (String word in words) {
            ans.Add(root.query(word));
        }
        return ans.ToArray();
    }
}
// 2410. Maximum Matching of Players With Trainers
/*Solution: Sort + Two Pointers
Sort players and trainers.

Loop through players, skip trainers until he/she can match the current players.

Time complexity: O(nlogn + mlogm + n + m)
Space complexity: O(1)*/
public class Solution {
    public int MatchPlayersAndTrainers(int[] players, int[] trainers) {
          Array.Sort(players);
    Array.Sort(trainers);
    int n = players.Length;
    int m = trainers.Length;
    int ans = 0;
    for (int i = 0, j = 0; i < n && j < m; ++i) {
      while (j < m && players[i] > trainers[j]) ++j;
      if (j++ == m) break;
      ++ans;
    }
    return ans;
    }
}
// 2409. Count Days Spent Together
/*Solution: Math
Convert date to days of the year.

Time complexity: O(1)
Space complexity: O(1)*/
public class Solution {
    public int CountDaysTogether(string arriveAlice, string leaveAlice, string arriveBob, string leaveBob) {
        int[] days = new int[12]{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    
    int getDay(string date) {// convert date to nth day of year, (1st-365th day)
         int idx = 0;
      int mm = (date[0] - '0') * 10 + (date[1] - '0');
      int dd = (date[3] - '0') * 10 + (date[4] - '0');
    for(int i=1; i<mm; i++) idx += days[i-1]; // or use prefix sum
    return idx+dd;
      //return accumulate(begin(days), begin(days) + mm - 1, dd);
    };
    
    int s = Math.Max(getDay(arriveAlice), getDay(arriveBob));
    int e = Math.Min(getDay(leaveAlice), getDay(leaveBob));
    return e >= s ? e - s + 1 : 0;  // no overlap
    }
}
// 2407. Longest Increasing Subsequence II
/*Solution: DP + Segment Tree | Max range query
Let dp[i] := length of LIS end with number i.
dp[i] = 1 + max(dp[i-k:i])

Naive dp takes O(n*k) time which will cause TLE.

We can use segment tree to speed up the max range query to log(m), where m is the max value of the array.

Time complexity: O(n*logm)
Space complexity: O(m)

*/
public class Solution {
    public int LengthOfLIS(int[] nums, int k) {
        int n = nums.Max();
    int[] dp = new int[2 * (n + 1)];
    int query(int l, int r) {
      int ans = 0;
      for (l += n, r += n; l < r; l >>= 1, r >>= 1) {
        if ((l & 1)>0) ans = Math.Max(ans, dp[l++]);      
        if ((r & 1)>0) ans = Math.Max(ans, dp[--r]);
      }
      return ans;
    };
    void update(int i, int val) {
      dp[i += n] = val;
      while (i > 1) {
        i >>= 1;
        dp[i] = Math.Max(dp[i * 2], dp[i * 2 + 1]);
      }
    };        
    int ans = 0;
    foreach (int x in nums) {
      int cur = 1 + query(Math.Max(1, x - k), x);
      update(x, cur);
      ans = Math.Max(ans, cur);
    }
    return ans;
    }
}
// 2405. Optimal Partition of String
/*Solution: Greedy
Extend the cur string as long as possible unless a duplicate character occurs.

You can use hashtable / array or bitmask to mark 
whether a character has been seen so far.

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int PartitionString(string s) {
         int ans = 1;
        HashSet<char> st = new HashSet<char>();
        foreach (char c in s) {
 		  // Insert Till we find duplicate element.
            if(!st.Contains(c)){
                st.Add(c);
            }
            else{
			 // If we found duplicate char then increment count and clear set and start with new set.
                ans++;
                st.Clear();
                st.Add(c);
            }
        }
        return ans;
    }
}

public class Solution {
    public int PartitionString(string s) {
       int[] m = new int[26];
    int ans = 0;
    foreach (char c in s) {
      if (++m[c - 'a'] == 1) continue;
      Array.Fill(m, 0);
      m[c - 'a'] = 1;
      ++ans;
    }
    return ans + 1;
    }
}

public class Solution {
    public int PartitionString(string s) {
       int mask = 0;
    int ans = 0;
    foreach (char c in s) {
      if ((mask & (1 << (c - 'a')) ) > 0) {
        mask = 0;        
        ++ans;
      }
      mask |= 1 << (c - 'a');
    }
    return ans + 1;
    }
}
// 2404. Most Frequent Even Element
/*Solution: Hashtable
Use a hashtable to store the frequency of even numbers.

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int MostFrequentEven(int[] nums) {
        Dictionary<int, int> m = new Dictionary<int, int>();
    int ans = -1;
    int best = 0;
    foreach (int x in nums) {
      if ((x & 1) > 0) continue;
        m[x] = m.GetValueOrDefault(x,0);
      int cur = ++m[x];
      if (cur > best) {
        best = cur;
        ans = x;
      } else if (cur == best && x < ans) {
        ans = x;
      }
    }
    return ans;
    }
}
// 2399. Check Distances Between Same Letters
/*Solution: Hashtable
Use a hastable to store the index of first occurrence of each letter.

Time complexity: O(n)
Space complexity: O(26)*/
public class Solution {
    public bool CheckDistances(string s, int[] distance) {
        int[] m = new int[26]; Array.Fill(m,-1);
    for (int i = 0; i < s.Length; ++i) {
      int c = s[i] - 'a';
      if (m[c] >= 0 && i - m[c] - 1 != distance[c]) return false;
      m[c] = i;
    }
    return true;
    }
}
// 2396. Strictly Palindromic Number
/*Intuition
The condition is extreme hard to satisfy, think about it...
for every base b between 2 and n - 2...
4 is not strictly palindromic number
5 is not strictly palindromic number
..
then the bigger, the more impossible.
Just return false


Prove
4 = 100 (base 2), so 4 is not strictly palindromic number
for n > 4, consider the base n - 2.
In base n - 1, n = 11.
In base n - 2, n = 12, so n is not strictly palindromic number.

There is no strictly palindromic number n where n >= 4


More
I think it may make some sense to ask if there a base b
between 2 and n - 2 that n is palindromic,
otherwise why it bothers to mention n - 2?

It's n - 2, not n - 1,
since for all n > 1,
n is 11 in base n - 2.
(Because n = (n - 1) + (1))

Then it's at least a algorithme problem to solve,
instead of a brain-teaser.

Maybe Leetcode just gave a wrong description.

Complexity
Time O(1)
Space O(1)*/
public class Solution {
    public bool IsStrictlyPalindromic(int n) {
        return false;
    }
}
// 2395. Find Subarrays With Equal Sum
/*Solution: Hashset
Use a hashset to track all the sums seen so far.

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public bool FindSubarrays(int[] nums) {
        HashSet<int> s = new HashSet<int>();
    for (int i = 1; i < nums.Length; ++i)
      if (!s.Add(nums[i] + nums[i - 1])) return true;    
    return false;
    }
}
// 2389. Longest Subsequence With Limited Sum
/*Solution: Sort + PrefixSum + Binary Search
Time complexity: O(nlogn + mlogn)
Space complexity: O(1)*/
public class Solution {
    public int[] AnswerQueries(int[] nums, int[] queries) {
        Array.Sort(nums);
        for (int i = 1; i < nums.Length; i++)
        {
            nums[i] = nums[i] + nums[i - 1];
        }


        List<int> ans = new List<int>();
        foreach (int q in queries)
          ans.Add(upperBound(nums, q) - 0);
        return ans.ToArray();
    }
    
    public int upperBound(int[] nums, int target ) {
    int l = 0, r = nums.Length - 1;

    while( l <= r ){
          int m = (l + r) / 2;
      if (nums[m] > target) {
        r = m - 1;
      } else {
        l = m + 1;
      }
    }

    return l;
  }
}

// 2316. Count Unreachable Pairs of Nodes in an Undirected Graph
/*
Solution 1: DFS
Use DFS to find all CCs

Time complexity: O(V+E)
Space complexity: O(V+E)
*/
public class Solution {
    public long CountPairs(int n, int[][] edges) {
    List<int>[] g = new List<int>[n];
    for (var i = 0; i < n; i++) g[i] = new List<int>();
    foreach (int[] e in edges) {
          g[e[0]].Add(e[1]);
          g[e[1]].Add(e[0]);
    }
    int[] seen = new int[n]; Array.Fill(seen, 0);
    long cur = 0;
    
    void dfs(int u) {
      ++cur;
      foreach (int v in g[u]){
          if (seen[v]++ == 0) dfs(v); 
      }
            
    };
    long ans = 0;    
    for (int i = 0; i < n; ++i) {
      if (seen[i]++ > 0 ) continue;
    
      cur = 0;
      dfs(i);
      ans += (n - cur) * cur;
    }
    return ans / 2;
    }
}
/*
Solution 2: Union Find
Time complexity: O(V+E)
Space complexity: O(V)
*/
public class Solution {
    public long CountPairs(int n, int[][] edges) {
        int[] parents = Enumerable.Range(0, n - 0 + 1).ToArray();
        int[] counts = new int[n]; Array.Fill(counts, 1);
    
    int find(int x) {
      if (parents[x] == x) return x;
      return parents[x] = find(parents[x]);
    };
    
    foreach (int[] e in edges) {
      int ru = find(e[0]);
      int rv = find(e[1]);
      if (ru != rv) {
        parents[rv] = ru;
        counts[ru] += counts[rv];        
      }
    }
    long ans = 0;    
    for (int i = 0; i < n; ++i)      
      ans += n - counts[find(i)];
    return ans / 2;
    }
}

// 2317. Maximum XOR After Operations
/*
Solution: Bitwise OR
The maximum possible number MAX = nums[0] | nums[1] | … | nums[n – 1].

We need to prove:
1) MAX is achievable.
2) MAX is the largest number we can get.

nums[i] AND (nums[i] XOR x) means that we can turn any 1 bits to 0 for nums[i].

1) If the i-th bit of MAX is 1, which means there are at least one number with i-th bit equals to 1, 
however, for XOR, if there are even numbers with i-th bit equal to one, 
the final results will be 0 for i-th bit, we get a smaller number. By using the operation, 
we can choose one of them and flip the bit.

**1** XOR **1** XOR **1** XOR **1** = **0** =>
**0** XOR **1** XOR **1** XOR **1** = **1**

2) If the i-th bit of MAX is 0, which means the i-th bit of all the numbers is 0, 
there is nothing we can do with the operation, and the XOR will be 0 as well.
e.g. **0** XOR **0** XOR **0** XOR **0** = **0**

Time complexity: O(n)
Space complexity: O(1)*/

public class Solution {
    public int MaximumXOR(int[] nums) {
        int res = 0;
        foreach (int a in nums)
            res |= a;
        return res;
    }
}
// 133. Clone Graph
/*
Solution: DFS + Hashtable
Time complexity: O(V+E)
Space complexity: O(V+E)
*/
/*
// Definition for a Node.
public class Node {
    public int val;
    public IList<Node> neighbors;

    public Node() {
        val = 0;
        neighbors = new List<Node>();
    }

    public Node(int _val) {
        val = _val;
        neighbors = new List<Node>();
    }

    public Node(int _val, List<Node> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
}
*/

public class Solution {
    public Node CloneGraph(Node node) {
          if (node == null) return null;
    Dictionary<Node, Node> m = new Dictionary<Node, Node>();
    void dfs(Node u) {
      m[u] = new Node(u.val);
      foreach (Node v in u.neighbors) {
        if (!m.ContainsKey(v)) dfs(v);
        m[u].neighbors.Add(m[v]);
      }
    };
    dfs(node);
    return m[node];
    }
}
// 2373. Largest Local Values in a Matrix
/*
Solution: Brute Force
Time complexity: O(n*n*9)
Space complexity: O(n*n)*/
public class Solution {
    public int[][] LargestLocal(int[][] grid) {
        int m = grid.Length - 2;
    int[][] ans = new int[m][];
    for (int i = 0; i < m; ++i) {
        ans[i] = new int[m];
        for (int j = 0; j < m; ++j)        
            for (int dy = 0; dy <= 2; ++dy)
                for (int dx = 0; dx <= 2; ++dx)
                    ans[i][j] = Math.Max(ans[i][j], grid[i + dy][j + dx]);  
    }   
            
    return ans;
    }
}
//2374. Node With Highest Edge Score
public class Solution {
    public int EdgeScore(int[] edges) {
      int n = edges.Length;
      long[] s = new long[n];
      for (int i = 0; i < n; ++i)
        s[edges[i]] += i;
      return Array.IndexOf(s, s.Max());
    }
}
// 2265. Count Nodes Equal to Average of Subtree
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    int res = 0;
    public int AverageOfSubtree(TreeNode root) {
         // Returns {sum(val), sum(node), ans}
    dfs(root);
        return res;
    }
    
    private int[] dfs(TreeNode node) {
        if(node == null) {
            return new int[] {0,0};
        }
        
        int[] left = dfs(node.left);
        int[] right = dfs(node.right);
        
        int currSum = left[0] + right[0] + node.val;
        int currCount = left[1] + right[1] + 1;
        
        if(currSum / currCount == node.val) {
            res++;
        }
            
        return new int[] {currSum, currCount};
    }
}

// 2260. Minimum Consecutive Cards to Pick Up
/*
Solution: Hashtable
Record the last position of each number,
ans = min{cardi – last[cardi]}

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int MinimumCardPickup(int[] cards) {
        int n = cards.Length;
    Dictionary<int, int> m = new Dictionary<int, int>();
    int ans = Int32.MaxValue;
    for (int i = 0; i < n; ++i) {
      if (m.ContainsKey(cards[i]))
        ans = Math.Min(ans, i - m[cards[i]] + 1);
      m[cards[i]] = i;
    }
    return ans == Int32.MaxValue ? -1 : ans;
    }
}

// 2264. Largest 3-Same-Digit Number in String
/*
Solution:

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public string LargestGoodInteger(string num) {
        string ans = String.Empty;
    for (int i = 0; i < num.Length - 2; ++i) {
      if (num[i] == num[i + 1] && num[i] == num[i + 2] && 
         (ans.Length == 0 || num[i] > ans[0]))
        ans = num.Substring(i, 3);
    }
    return ans;
    }
}

// 2266. Count Number of Texts
// 2267. Check if There Is a Valid Parentheses String Path
// 2304. Minimum Path Cost in a Grid
/*
Solution: DP
Let dp[i][j] := min cost to reach grid[i][j] from the first row.

dp[i][j] = min{grid[i][j] + dp[i – 1][k] + moveCost[grid[i – 1][k]][j]} 0 <= k < n

For each node, try all possible nodes from the previous row.

Time complexity: O(m*n2)
Space complexity: O(m*n) -> O(n)*/
public class Solution {
    public int MinPathCost(int[][] grid, int[][] moveCost) {
       
    int m = grid.Length;
    int n = grid[0].Length;
    int[][] dp = new int[m][]; //(m, vector<int>(n, INT_MAX));    
    for (int i = 0; i < m; ++i){
        dp[i] = new int[n]; Array.Fill(dp[i],Int32.MaxValue);
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)          
                dp[i][j] = Math.Min(dp[i][j], grid[i][j] + 
                         (i > 0 ? dp[i - 1][k] + moveCost[grid[i - 1][k]][j] : 0));
    }
      
    return dp[m - 1].Min();
    }
}

// 2303. Calculate Amount Paid in Taxes
/*
Solution: Follow the rules
“Nothing is certain except death and taxes” – Benjamin Franklin

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public double CalculateTax(int[][] brackets, int income) {
          double ans = 0;
          for (int i = 0; i < brackets.Length; ++i) {      
            ans += (Math.Min(income, brackets[i][0]) - (i > 0 ? brackets[i-1][0] : 0)) * brackets[i][1] / 100.0;      
            if (brackets[i][0] >= income) break;
          }
          return ans;
    }
}
// 2315. Count Asterisks
/*
Explanation
Parse the input, if currently met odd bars, we count *.

Complexity
Time O(n)
Space O(1)*/
public class Solution {
    public int CountAsterisks(string s) {
        int res = 0, bars = 0;
        for (int i = 0; i < s.Length; ++i) {
            if (s[i] == '*' && bars % 2 == 0)
                res++;
            if (s[i] == '|')
                bars++;
        }
        return res;
    }
}
/*
Solution: Counting
Count the number of bars so far, 
and only count ‘*’ when there are even number of bars on the left.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int CountAsterisks(string s) {
        int bars = 0;
        int ans = 0;    
        foreach (char c in s) {
          if (c == '*' && bars % 2 == 0)
            ++ans;
          if (c == '|') ++bars;      
        }
        return ans;
    }
}
// 2269. Find the K-Beauty of a Number
/*
Solution: Substring
Note: the substring can be 0, e.g. “00”

Time complexity: O((l-k)*k)
Space complexity: O(l + k) -> O(1)*/
public class Solution {
    public int DivisorSubstrings(int num, int k) {
        string s = num.ToString();
        int ans = 0;
        for (int i = 0; i <= s.Length - k; ++i) {
          int t = Int32.Parse(s.Substring(i, k));
          ans += Convert.ToInt32(t > 0 && (num % t == 0));
        }
        return ans;
      
    }
}
// 2270. Number of Ways to Split Array
/*
Solution: Prefix/Suffix Sum
Note: sum can be greater than 2^31, use long!

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int WaysToSplitArray(int[] nums) {
        int ans = 0; 
        long l = 0;// long r = nums.Sum();// Sum() cause stackoverflow!
        long r = 0; Array.ForEach(nums, i => r += i);
        //long r = nums.Aggregate((total, next) => total + next); got wrong answer!
        for (long i = 0; i < nums.Length - 1; ++i)       
          ans += Convert.ToInt32((l += nums[i]) >= (r -= nums[i]));
        return ans;
    }
}

// 2251. Number of Flowers in Full Bloom
/*
Solution 1: Binary Seach
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
Space O(n)*/

public class Solution {
    public int[] FullBloomFlowers(int[][] flowers, int[] persons) {
        List<int> start = new List<int>(), end = new List<int>();
        foreach (var f in flowers){
             start.Add(f[0]); end.Add(f[1]);
        }
           
        start.Sort((x, y) => (x.CompareTo(y)));
        end.Sort((x, y) => (x.CompareTo(y)));
        List<int> res = new List<int>();
        foreach (int t in persons) {
            int started = upperBound(start, t);
            int ended = lowerBound(end, t);
            res.Add(started - ended);
        }
        return res.ToArray();
       
    }
    
    public int lowerBound(List<int> nums, int target ) {
      int l = 0, r = nums.Count - 1;

      while (l <= r) {
            int m = (l + r) / 2;
        if (nums[m] >= target) {
          r = m - 1;
        } else {
          l = m + 1;
        }
      }

      return l;
    }

  public int upperBound(List<int> nums, int target ) {
    int l = 0, r = nums.Count - 1;

    while( l <= r ){
          int m = (l + r) / 2;
      if (nums[m] > target) {
        r = m - 1;
      } else {
        l = m + 1;
      }
    }

    return l;
  }
}
//TLE
/*
Solution: Prefix Sum + Binary Search
Use a treemap to store the counts (ordered by time t), 
when a flower begins to bloom at start, we increase m[start], 
when it dies at end, we decrease m[end+1]. 
prefix_sum[t] indicates the // of blooming flowers at time t.

For each people, use binary search to find the latest // of flowers 
before his arrival.

Time complexity: O(nlogn + mlogn)
Space complexity: O(n)
*/
public class Solution {
    public int[] FullBloomFlowers(int[][] flowers, int[] persons) {
        SortedDictionary<int, int> m = new SortedDictionary<int, int>{{0,0}};
        foreach (int[] f in flowers){
            if (!m.ContainsKey(f[0])) m.Add(f[0],1);
            else m[f[0]]+=1;
            if (!m.ContainsKey(f[1]+1)) m.Add(f[1]+1,-1);
            else m[f[1]+1]-=1; 
        }
        int sum = 0;
       
        for(int i = 0; i< m.Count; i++)
        {
            sum += m.ElementAt(i).Value;
            int k = m.ElementAt(i).Key;
            m.Remove(k);
            m.Add(k,sum);
            
        }
        
        List<int> ans = new List<int>();
        foreach (int p in persons) {
            int it = upperBound(m.Keys.ToList(), p) ;
            int c = it-1;
           ans.Add(m.ElementAt(c).Value);
        }
        
        return ans.ToArray();
        
        
    }
    public int upperBound(List<int> nums, int target ) {
      int l = 0, r = nums.Count - 1;

      while( l <= r ){
            int m = (l + r) / 2;
        if (nums[m] > target) {
          r = m - 1;
        } else {
          l = m + 1;
        }
      }

      return l;
    }
}
// 2255. Count Prefixes of a Given String
/*
Explanation
for each word w in words list,
check if word w startsWith the string s


Complexity
Time O(NS)
Space O(1)
*/
public class Solution {
    public int CountPrefixes(string[] words, string s) {
        int res = 0;
        foreach (String w in words)
            if (s.StartsWith(w))
                res++;
        return res;
    }
}

// 2256. Minimum Average Difference
public class Solution {
    public int MinimumAverageDifference(int[] nums) {
        int n = nums.Length;
        long suffix =0; Array.ForEach(nums, i => suffix += i);
        long prefix = 0;
        long best = Int32.MaxValue;
        int ans = 0;
        for (int i = 0; i < n; ++i) {
          prefix += nums[i];
          suffix -= nums[i];
          long cur = Math.Abs(prefix / (i + 1) - (i == n - 1 ? 0 : suffix / (n - i - 1)));
          if (cur < best) {
            best = cur;
            ans = i;
          }
        }
        return ans;
    }
}

// 2257. Count Unguarded Cells in the Grid
public class Solution {
    public int CountUnguarded(int m, int n, int[][] guards, int[][] walls) {
        int[][] s = new int[m][]; 
        for(int i = 0; i < m; i++) s[i] = new int[n];
      foreach (int[] g in guards)
        s[g[0]][g[1]] = 2;
      
      foreach (int[] w in walls)
        s[w[0]][w[1]] = 3;
      
      for (int i = 0; i < m; ++i){
          for (int j = 0; j < n; ++j) {
          if (s[i][j] != 2) continue;
          for (int y = i - 1; y >= 0; s[y--][j] = 1)
            if (s[y][j] >= 2) break;          
          for (int y = i + 1; y < m; s[y++][j] = 1)
            if (s[y][j] >= 2) break;        
          for (int x = j - 1; x >= 0; s[i][x--] = 1)
            if (s[i][x] >= 2) break;    
          for (int x = j + 1; x < n; s[i][x++] = 1)
            if (s[i][x] >= 2) break;      
      }
            
        }    
      int ans = 0;
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
          ans += Convert.ToInt32(s[i][j] == 0);
      return ans;
    }
}

// 2259. Remove Digit From Number to Maximize Result
public class Solution {
    public string RemoveDigit(string number, char digit) {
         int n = number.Length;
        string ans = new String('1', n - 1);
        for (int i = 0; i < n; ++i)
          if (number[i] == digit)
            ans = BigInteger.Parse(ans) > BigInteger.Parse(number.Substring(0, i) + number.Substring(i + 1)) ? ans : number.Substring(0, i) + number.Substring(i + 1);
        return ans;
    }
}

// 2242. Maximum Score of a Node Sequence
/*
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
Space O(n + m)*/
public class Solution {
    public int MaximumScore(int[] A, int[][] edges) {
      int n = A.Length;
        List<int>[] q = new List<int>[n];
        for (int i = 0; i < n; i++)
            q[i] = new List<int>();
       
        foreach (int[] e in edges) {
            q[e[0]].Add(e[1]);
            q[e[1]].Add(e[0]);
            // q[e[0]].Sort((a, b) => A[a] - A[b]);  works too!
              //q[e[1]].Sort((a, b) => A[a] - A[b]);  works too!
            q[e[0]] = q[e[0]].OrderBy(a => A[a]).ToList(); // Orderby works in this way
            q[e[1]] = q[e[1]].OrderBy(a => A[a]).ToList(); // list = list.OrderBy().ToList()
            if (q[e[0]].Count > 3) q[e[0]].RemoveAt(0);
            if (q[e[1]].Count > 3) q[e[1]].RemoveAt(0);

        }
      
        int res = -1;
        foreach (int[] edge in edges)
            foreach (int i in q[edge[0]])
                foreach (int j in q[edge[1]])
                    if (i != j && i != edge[1] && j != edge[0])
                        res = Math.Max(res, A[i] + A[j] + A[edge[0]] + A[edge[1]]);
                
        return res;
    }
}
// 2243. Calculate Digit Sum of a String
public class Solution {
    public string DigitSum(string s, int k) {
        while (s.Length > k) {
      string ss = String.Empty;
      for (int j = 0; j < s.Length; j += k) {
        int sum = 0;
        for (int i = 0; i < k && i + j < s.Length; ++i)
          sum += (s[j + i] - '0');
        ss += sum.ToString();
      }
        string temp = ss;
        ss = s;
        s = temp;
    }
    return s;  
    }
}
// 2244. Minimum Rounds to Complete All Tasks
/*Intuition
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
Space O(n)

Solution: Math
Count the frequency of each level. The only case that can not be finished is 1 task at some level. Otherwise we can always finish it by 2, 3 tasks at a time.

if n = 2: 2 => 1 round
if n = 3: 3 => 1 round
if n = 4: 2 + 2 => 2 rounds
if n = 5: 3 + 2 => 2 rounds
…
if n = 3k, n % 3 == 0 : 3 + 3 + … + 3 = k rounds
if n = 3k + 1, n % 3 == 1 : 3*(k – 1) + 2 + 2 = k + 1 rounds
if n = 3k + 2, n % 3 == 2 : 3*k + 2 = k + 1 rounds

We need (n + 2) / 3 rounds.

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int MinimumRounds(int[] tasks) {
        Dictionary<int, int> count = new Dictionary<int, int>();
        foreach (int a in tasks)
            count[a] = count.GetValueOrDefault(a, 0) + 1;
        int res = 0;
        foreach (int freq in count.Values) {
            if (freq == 1) return -1;
            res += (freq + 2) / 3;
        }
        return res;
        
    }
}
// 2248. Intersection of Multiple Arrays
/*
Solution: Hashtable
Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public IList<int> Intersection(int[][] nums) {
        int[] m = new int[1001];
    foreach (int[] arr in nums)
      foreach (int x in arr)
        ++m[x];
    IList<int> ans = new List<int>();
    for (int i = 0; i < m.Length; ++i)
      if (m[i] == nums.Length)
        ans.Add(i);
    return ans;
    }
}
// 2249. Count Lattice Points Inside a Circle
/*
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
*/
public class Solution {
    public int CountLatticePoints(int[][] circles) {
        HashSet<int> res = new HashSet<int>();
        foreach (int[] c in circles)
            for (int i = -c[2]; i <= c[2]; i++)
                for (int j = -c[2]; j <= c[2]; j++)
                    if (i * i + j * j <= c[2] * c[2])
                        res.Add((c[0] + i) * 1000 + c[1] + j);
        return res.Count;
    }
}
/*
Solution 2: Check all points, O(1) space
Explanation
For each point (i, j) in the plan,
enumerate i from 0 to 200
enumerate j from 0 to 200
Check if (i, j) in any circle.


Complexity
Same time complexity, O(1) space.

Time O(NXY)
Space O(1)
where X = 100 is the range of xi
where Y = 100 is the range of yi
*/
public class Solution {
    public int CountLatticePoints(int[][] circles) {
        int res = 0;
        for (int i = 0; i <= 200; i++) {
            for (int j = 0; j <= 200; j++) {
                foreach (int[] c in circles) {
                    if ((c[0] - i) * (c[0] - i) + (c[1] - j) * (c[1] - j) <= c[2] * c[2]) {
                        res++;
                        break;
                    }
                }
            }
        }
        return res;
    }
}
/*
Solution: Brute Force
Time complexity: O(m * m * n) = O(200 * 200 * 200)
Space complexity: O(1)
*/
public class Solution {
    public int CountLatticePoints(int[][] circles) {
        bool isIn(int u, int v, int x, int y, int r) {
      return (u - x) * (u - x) + (v - y) * (v - y) <= r * r;
    };    
    int ans = 0;
    for (int u = 0; u <= 200; ++u)
      for (int v = 0; v <= 200; ++v) {
        bool found = false;
        foreach (var c in circles)
          if (isIn(u, v, c[0], c[1], c[2])) {
            found = true;
            break;
          }
        ans += Convert.ToInt32(found);
      }
    return ans;
    }
}

// 2241. Design an ATM Machine
/*Solution:
Follow the rules. Note: total count can be very large, use long instead.

Time complexity: O(1)
Space complexity: O(1)*/
public class ATM {

    long[] notes = new long[]{20, 50, 100, 200, 500}; 
    long[] counts;
    public ATM() {
        
        counts = new long[5];
    }
    
    public void Deposit(int[] banknotesCount) {
        for(int i = 0; i < banknotesCount.Length; i++)       
        {
            counts[i] += banknotesCount[i];
        }
    }
    
    public int[] Withdraw(int amount) {
       int[] ans = new int[5];
        long[] tmp = new long[5];
        //tmp = counts; won't work!
        Array.Copy(counts, tmp, 5);// Array.Copy() works!
        // or looping through array copy each item
        /*for(int i = 0; i < tmp.Length; i++)
        {
            tmp[i] = counts[i];
        }*/
        
        for(int i = counts.Length - 1; i >= 0; i--)
        {
            if(amount >= notes[i])
            {
                int c = (int)Math.Min(counts[i], amount / notes[i]);                
                ans[i] = c;                
                amount -= Convert.ToInt32(c * notes[i]);
                tmp[i] -= c;
            }
        }    
        
        if(amount != 0) return new int[]{-1};
      
        counts = tmp;
        return ans;
    }
   
}

/**
 * Your ATM object will be instantiated and called as such:
 * ATM obj = new ATM();
 * obj.Deposit(banknotesCount);
 * int[] param_2 = obj.Withdraw(amount);
 */

// 2240. Number of Ways to Buy Pens and Pencils
/*
Solution:
Enumerate all possible ways to buy k pens, e.g. 0 pen, 1 pen, …, total / cost1.
The way to buy pencils are (total – k * cost1) / cost2 + 1.
ans = sum((total – k * cost1) / cost2 + 1)) for k = 0 to total / cost1.

Time complexity: O(total / cost1)
Space complexity: O(1)*/
public class Solution {
    public long WaysToBuyPensPencils(int total, int cost1, int cost2) {
     long ans = 0;
    for (long k = 0; k <= total / cost1; ++k)
      ans += ((total - k * cost1) / cost2) + 1;
    return ans;
    }
}

// 2239. Find Closest Number to Zero

// 2236. Root Equals Sum of Children 
/*Solution:
Just want to check whether you know binary tree or not.

Time complexity: O(1)
Space complexity: O(1)*/
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public bool CheckTree(TreeNode root) {
         return root.val == root.left.val + root.right.val;
    }
}

// 2235. Add Two Integers
/*
Solution: Just sum them up
Time complexity: O(1)
Space complexity: O(1)*/
public class Solution {
    public int Sum(int num1, int num2) {
        return num1 + num2;
    }
}

// 2233. Maximum Product After K Increments
public class Solution {
    public int MaximumProduct(int[] nums, int k) {
        double kMod = 1e9 + 7;
    PriorityQueue<int, int> q = new PriorityQueue<int, int>();
    foreach(var item in nums){
            q.Enqueue(item,item);
        }
    while (k-- > 0) {
      int n = q.Peek(); 
      q.Dequeue();
      q.Enqueue(n+1, n+1);
    }
    long ans = 1;
    while (q.Count != 0) {
      ans *= q.Peek(); q.Dequeue();
      ans %= Convert.ToInt64(kMod);
    }
    return Convert.ToInt32(ans);
    }
}

public class Solution {
    public int MaximumProduct(int[] nums, int k) {
        double kMod = 1e9 + 7;
    PriorityQueue<int, int> q = new PriorityQueue<int, int>(Comparer<int>.Create((a, b) => a - b)); // find greater at front
    foreach(var item in nums){
            q.Enqueue(item,item);
        }
    while (k-- > 0) {
      int n = q.Peek(); 
      q.Dequeue();
      q.Enqueue(n+1, n+1);
    }
    long ans = 1;
    while (q.Count != 0) {
      ans *= q.Peek(); q.Dequeue();
      ans %= Convert.ToInt64(kMod);
    }
    return Convert.ToInt32(ans);
    }
}

// 2232. Minimize Result by Adding Parentheses to Expression
public class Solution {
    public string MinimizeResult(string expression) {
       int n = expression.Length;
    int p = expression.IndexOf('+');
    int best = Int32.MaxValue;
    string ans = String.Empty;
    for (int l = 0; l < p; l++){
         for (int r = p + 2; r < n + 1; r++) {             
         int m1 = (l > 0) ? int.Parse(expression.Substring(0, l)) : 1;
         int m2 = (r < n) ? int.Parse(expression.Substring(r)) : 1;    
         int n1 = int.Parse(expression.Substring(l, p-l));
        // substring second parameter is length!
        int n2 = int.Parse(expression.Substring(p + 1, r - p - 1 ));
         int cur = m1 * (n1 + n2) * m2;
        if (cur < best) {
          best = cur;
         //ans = expression.Insert(l, "(").Insert(r + 1, ")"); 
         //above line is less space but slower
        // below works too!! quicker! would need more space!
            ans = expression;
            ans = ans.Insert(l, "("); // when use insert make sure like this
           ans = ans.Insert(r + 1, ")"); // str = str.Insert(); str.Insert() won't work!
        }
      }
    }
    return ans;
    }
}

public class Solution {
    public string MinimizeResult(string expression) {
        var arr = expression.Split("+");
            int res = int.MaxValue;
            int x = 0;//"(" always insert at the left side of every digits in left-int-string
            int y = 0;//")" always insert at the right side of every digits in right-int-string
            for (int i = 0; i < arr[0].Length; i++)
            {
                for (int j = 0; j < arr[1].Length; j++)
                {
				    // calculation of expression is val1*val2*val3, val2 is "(a+b)"
				    //if insert at the 0 index, first part shoule be 1
                    int val1 = i == 0 ? 1 : int.Parse(arr[0].Substring(0, i));
                    int val2 = int.Parse(arr[0].Substring(i)) + int.Parse(arr[1].Substring(0, j + 1));
                    int val3 = j == arr[1].Length - 1 ? 1 : int.Parse(arr[1].Substring(j + 1));
                    if (val1 * val2 * val3 < res)
                    {
                        res = val1 * val2 * val3;
                        x = i;
                        y = j;
                    }
                }
            }
			// "(" alwasy insert at left of a digit, so x is the correct insert index
			//but ")" insert at right of a digit, so need y+1 to get the correct index 
            return arr[0].Insert(x,"(")+"+"+arr[1].Insert(y+1,")");
    }
}

// 2231. Largest Number After Digit Swaps by Parity
public class Solution {
    public int LargestInteger(int num) {
        List<int> odd = new List<int>(), even = new List<int>(), org = new List<int>();
    for (int cur = num; cur > 0; cur /= 10) {
      (((cur & 1) > 0) ? odd : even).Add(cur % 10);// use like this is really simple
      org.Add(cur % 10);
    }
    odd.Sort();
    even.Sort();  
    int ans = 0;
    for (int i = 0; i < org.Count; ++i) {
      if (i > 0) ans *= 10;
      List<int> cur = ((org[org.Count - i - 1] & 1) > 0) ? odd : even;
      ans += cur[cur.Count - 1];
      cur.RemoveAt(cur.Count - 1);// make sure is RemoveAt
    }
    return ans;
    }
}

// 2227. Encrypt and Decrypt Strings
public class Encrypter {

    private string[] vals = new string[26];  
    private string[] dict;
    public Encrypter(char[] keys, string[] values, string[] dictionary) {
        dict = dictionary;
        for (int i = 0; i < keys.Length; ++i)
            vals[keys[i] - 'a'] = values[i]; 
    }
    
    public string Encrypt(string word1) {
         string ans = String.Empty;
        foreach (char c in word1){
            if (vals[c - 'a'] == null ) return "";
            ans += vals[c - 'a'];
        }
        return ans;
    }
    
    public int Decrypt(string word2) {
        /*return count_if(begin(dict), end(dict), [&](const string& w){ 
      return Encrypt(w) == word2;
    });*/
        int count = 0;
        foreach(string s in dict){
            if (Encrypt(s) == word2) {
                count += 1;
            }
        }
        return count;
    }
}

/**
 * Your Encrypter object will be instantiated and called as such:
 * Encrypter obj = new Encrypter(keys, values, dictionary);
 * string param_1 = obj.Encrypt(word1);
 * int param_2 = obj.Decrypt(word2);
 */

/*
Explanation
The hashmap enc help binding each paire of keys[i] and values[i],
so that we can encrypt a char to the string in O(1)

count counts the frequency of words in dictionary after encrypt,
then we can used in decrypt in O(1).


Complexity
Encrypter Time O(n) Space O(n)
encrypt Time O(word1) Space O(word1)
decrypt Time O(1) Space O(1)


Note
Not all word can be "encrypt",
For character c, if we can't find the index i satisfying keys[i] == c in keys.
The behavior are NOT clearly defined.

In my opinion we should do nothing but keep the original character,
(the standard solution of OJ doesn't work as I suggest)

These kind of test cases are not present in the original test cases set,
but recedntly blindly added to the test cases.

The descrption of probelm should be fixed, not blindly add an appropriat test cases.

It's like, a bug is reported and not guarded by tests,
then LC adds a test but not fix anything at all.*/
 public class Encrypter {

    Dictionary<char, string> enc;
    Dictionary<string, int> count;
    public Encrypter(char[] keys, string[] values, string[] dictionary) {
        enc = new Dictionary<char, string>();
        for (int i = 0; i < keys.Length; ++i)
            enc[keys[i]] = values[i];
        
        count = new Dictionary<string, int>();
        foreach (String w in dictionary) {
            String e = Encrypt(w);
            count[e] = count.GetValueOrDefault(e, 0) + 1;
        }
    }
    
    public string Encrypt(string word1) {
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < word1.Length; ++i)
            res.Append(enc.GetValueOrDefault(word1[i], "#"));
        return res.ToString();
    }
    
    public int Decrypt(string word2) {
        return count.GetValueOrDefault(word2, 0);
    }
}

/**
 * Your Encrypter object will be instantiated and called as such:
 * Encrypter obj = new Encrypter(keys, values, dictionary);
 * string param_1 = obj.Encrypt(word1);
 * int param_2 = obj.Decrypt(word2);
 */

 // 2226. Maximum Candies Allocated to K Children
public class Solution {
    public int MaximumCandies(int[] candies, long k) {
         long total = 0; Array.ForEach(candies, i => total += i);
    long l = 1;
    long r = total / k + 1;
    while (l < r) {
      long m = l + (r - l) / 2;
      long c = 0;
      foreach (int pile in candies)
        c += pile / m;
      if (c < k)
        r = m;
      else
        l = m + 1;
    }
    return Convert.ToInt32(l - 1);
    }
}
/*Intuition
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
Space O(1)*/
public class Solution {
    public int MaximumCandies(int[] candies, long k) {
        int left = 0, right = 10_000_000;
        while (left < right) {
            long sum = 0;
            int mid = (left + right + 1) / 2;
            foreach (int a in candies) {
                sum += a / mid;
            }
            if (k > sum)
                right = mid - 1;
            else
                left = mid;
        }
        return left;
    }
}

// 2239. Find Closest Number to Zero
public class Solution {
    public int FindClosestNumber(int[] nums) {
       /* return *min_element(begin(nums), end(nums), [](int a, int b){
      return abs(a) < abs(b) || (abs(a) == abs(b) && a > b
                      
    });*/
        Array.Sort(nums, (x, y) => Math.Abs(x) == Math.Abs(y) ? y - x :  Math.Abs(x) - Math.Abs(y));
       
        return nums[0]; // min value
    }
}
// 2225. Find Players With Zero or One Losses
// SortedDictionary
public class Solution {
    public IList<IList<int>> FindWinners(int[][] matches) {
        SortedDictionary<int, int> m = new SortedDictionary<int, int>();
    foreach (int[] match in matches) {
      m[match[0]] = m.GetValueOrDefault(match[0], 0);
      m[match[1]]= m.GetValueOrDefault(match[1], 0)+1;      
    }
    IList<IList<int>> ans = new List<IList<int>>();
     for( int i = 0; i<2; i++)
        ans.Add(new List<int>());

    foreach ( var kv in m){
        if (kv.Value <= 1) {
            ans[kv.Value].Add(kv.Key);  
        }
    }

    return ans;
    }
}
// Linq
public class Solution {
    public IList<IList<int>> FindWinners(int[][] matches) {
        var result = new List<IList<int>>();
            var winners = matches.Select((x, i) => matches[i][0]);
            var losers = matches.Select((x, i) => matches[i][1]);

            var neverLost = winners.Distinct().Except(losers);
            var lostOnce = losers.GroupBy(x => x).Where(g => g.Count() == 1).Select(y => y.Key);

            result.Add(neverLost.OrderBy(x => x).ToList());
            result.Add(lostOnce.OrderBy(x => x).ToList());
            return result;
    }
}
//Linq
public class Solution {
    public IList<IList<int>> FindWinners(int[][] matches) {
         var losses = matches.Select(x => x[1]);

        return new int[][]{
            
            matches
                .Select(x => x[0])
                .Except(losses)
                .OrderBy(x => x)
                .ToArray(),
            
            losses
                .GroupBy(x => x)
                .Where(g => g.Count() == 1)
                .Select(g => g.Key)
                .OrderBy(x => x)
                .ToArray()
        };
    }
}
// Dictionary 
/*Solution: Hashtable
Use a hashtable to store the number of matches each player has lost. Note, also create an entry for those winners who never lose.

Time complexity: O(m), m = # of matches
Space complexity: O(n), n = # of players

*/
public class Solution {
    public IList<IList<int>> FindWinners(int[][] matches) {
        Dictionary<int, int> m = new Dictionary<int, int>();
    foreach (int[] match in matches) {//doesn't check contains
      m[match[0]] = m.GetValueOrDefault(match[0], 0);
      m[match[1]]= m.GetValueOrDefault(match[1], 0)+1;      
    }
    IList<IList<int>> ans = new List<IList<int>>();
     for( int i = 0; i<2; i++)
        ans.Add(new List<int>());

    foreach ( var kv in m){
        if (kv.Value <= 1) {
            ans[kv.Value].Add(kv.Key);  
        }
    }     
        /* below doesn't work!  
        ans[0].ToList().Sort(); 
        ans[1].ToList().Sort(); 
        return ans;*/
        //OrderBy works better!
        ans[0] = ans[0].OrderBy(x => x).ToList();
        ans[1] = ans[1].OrderBy(x => x).ToList();
        return ans;
    }
}
// Dictionary 
public class Solution {
    public IList<IList<int>> FindWinners(int[][] matches) {
        Dictionary<int, int> d = new Dictionary<int, int>();
        for (int i = 0; i < matches.Length; i++) { //check contains
            if (!d.ContainsKey(matches[i][0])) d[matches[i][0]] = 0;
            if (!d.ContainsKey(matches[i][1])) d[matches[i][1]] = 0;
            d[matches[i][1]]++;
        }
        
        List<IList<int>> ans = new List<IList<int>>(capacity: 2);
        for (int i = 0; i < 2; i++) ans.Add(new List<int>());
        
        foreach (var (player, losses) in d) { // var (key, value) in dictionary
            if (losses < 2) ans[losses].Add(player);
        }
        //use OrderBy in list of lists
        for(int i = 0; i < 2; i++) ans[i] = ans[i].OrderBy(x => x).ToList();
        return ans;
    }
}
// 2224. Minimum Number of Operations to Convert Time
/*Solution: Greedy
Start with 60, then 15, 5 and finally increase 1 minute a time.

Time complexity: O(1)
Space complexity: O(1)

*/
public class Solution {
    public int ConvertTime(string current, string correct) {
        int getMinute (string t) {
      return (t[0] - '0') * 10 * 60 
             + (t[1] - '0') * 60 
             + (t[3] - '0') * 10 
             + (t[4] - '0');
    };
    
    int t1 = getMinute(current);
    int t2 = getMinute(correct);
    int ans = 0;    
    foreach (int d in new int[]{60, 15, 5, 1})
      while (t2 - t1 >= d) {
        ++ans;
        t1 += d;
      }
    return ans;
    }
}
// 2223. Sum of Scores of Built Strings
/*Solution: Z-Function
Time complexity: O(n)
Space complexity: O(n)

*/
public class Solution {
    public long SumScores(string s) {
        long zFunc (string s) {
      int n = s.Length;
      long[] z = new long[n];
            long sum = 0, l = 0, r = 0; 
      for (long i = 1; i < n; ++i) {
        if (i <= r)
          z[i] = Math.Min(r - i + 1, z[i - l]);
        while (i + z[i] < n && s[(int)z[i]] == s[(int)i + (int)z[i]])
          ++z[i];
        if (i + z[i] - 1 > r) {
          l = i;
          r = i + z[i] - 1;
        }      
      }
        Array.ForEach(z, i => sum += i);
      return sum;
    };    
    return zFunc(s) + s.Length;
    }
}
// 2222. Number of Ways to Select Buildings
/*Solution: DP
The brute force solution will take O(n3) which will lead to TLE.

Since the only two valid cases are “010” and “101”.

We just need to count how many 0s and 1s, 
thus we can count 01s and 10s and finally 010s and 101s.*/
public class Solution {
    public long NumberOfWays(string s) {
        long c0 = 0, c1 = 0, c01 = 0, c10 = 0, c101 = 0, c010 = 0;
    foreach (char c in s) {
      if (c == '0') {
        ++c0;
        c10 += c1;
        c010 += c01;
      } else {
        ++c1;
        c01 += c0;
        c101 += c10;
      }
    }
    return c101 + c010;
    }
}
// 2221. Find Triangular Sum of an Array
/*Solution 1: Simulation

Time complexity: O(n2)
Space complexity: O(n)

*/
public class Solution {
    public int TriangularSum(int[] nums) {
        while (nums.Length != 1) {
      int[] next = new int[nums.Length - 1];
      for (int i = 0; i < nums.Length - 1; ++i)
        next[i] = (nums[i] + nums[i + 1]) % 10;
      int[] tmp = next;
    next = nums;
    nums = tmp;
    }
    return nums[0];
    }
}
// 2220. Minimum Bit Flips to Convert Number
/*Solution: XOR

start ^ goal will give us the bitwise difference of start and goal in binary format.
ans = # of 1 ones in the xor-ed results.
For C++, we can use __builtin_popcount or bitset<32>::count() to get the number of bits set for a given integer.

Time complexity: O(1)
Space complexity: O(1)

We need to count the number of corresponding bits of start and goal that are different.
xor-ing start and goal will result in a new number with binary representation of 0 where the corresponding bits of start and goal are equal and 1 where the corresponding bits are different.

For example: 10 and 7
10 = 1010
7 = 0111

10 xor 7 = 1101 (3 ones)

Next we need to count the number of 1s (different bits)
The quickest way to count the number of 1s in a number is by eliminating the right most 1 each time and count the number of eliminations, this is done by and-ing the number with (number-1)
Subtracting a 1 from a number flips all right most bits until the first right most 1 and by and-ing with the number itself we eliminating the all bits until the first tight most 1 (inclusive)
ex.
number =1101
number -1 = 1100
number and (number -1) = 1100 (we eliminated the right most 1)
*/
public class Solution {
    public int MinBitFlips(int start, int goal) {
        int xor = start ^ goal;
        int count = 0;
        while( xor > 0){
            count++;
            xor = xor & (xor-1);
        }
        return count;
    }
}
// 2218. Maximum Value of K Coins From Piles

// 2217. Find Palindrome With Fixed Length
/*
Solution: Math
For even length e.g. 4, we work with length / 2, e.g. 2. Numbers: 10, 11, 12, …, 99, starting from 10, ends with 99, which consist of 99 – 10 + 1 = 90 numbers. For the x-th number, e.g. 88, the left part is 10 + 88 – 1 = 97, just mirror it o get the palindrome. 97|79. Thus we can answer a query in O(k/2) time which is critical.


For odd length e.g. 3 we work with length / 2 + 1, e.g. 2, Numbers: 10, 11, 12, 99. Drop the last digit and mirror the left part to get the palindrome. 101, 111, 121, …, 999.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public long[] KthPalindrome(int[] queries, int intLength) {
        long s = Convert.ToInt64(Math.Pow(10, (intLength + 1) / 2 - 1));
    long e = s * 10;
    long gen(long x, bool isOdd) {
      long ans = x;
      for (long c = isOdd == true ? (x / 10) : x; c > 0; c /= 10)
        ans = ans * 10 + c % 10;
      return ans;
    };
    List<long> ans = new List<long>();
    foreach (int q in queries)
      ans.Add(q > e - s ? -1 : gen(s + q - 1, Convert.ToBoolean(intLength & 1)));
    return ans.ToArray();
    }
}
// 2216. Minimum Deletions to Make Array Beautiful
/*
Solution: Greedy + Two Pointers
If two consecutive numbers are the same, we must remove one. We don’t need to actually remove elements from array, just need to track how many elements have been removed so far.

i is the position in the original array, ans is the number of elements been removed. i – ans is the position in the updated array.

ans += nums[i – ans] == nums[i – ans + 1]

Remove the last element (just increase answer by 1) if the length of the new array is odd.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MinDeletion(int[] nums) {
         int n = nums.Length;
    int ans = 0;
    for (int i = 0; i - ans + 1 < n; i += 2)
      ans += Convert.ToInt32(nums[i - ans] == nums[i - ans + 1]);
    ans += (n - ans) & 1;
    return ans;
    }
}
// 2215. Find the Difference of Two Arrays
/*Solution: Hashtable
Use two hashtables to store the unique numbers of array1 and array2 respectfully.

Time complexity: O(m+n)
Space complexity: O(m+n)*/
public class Solution {
    public IList<IList<int>> FindDifference(int[] nums1, int[] nums2) {
         IList<IList<int>> ans = new List<IList<int>>();
        for (int i = 0; i < 2; i++) ans.Add(new List<int>());
    int[] s1 = new HashSet<int>(nums1).ToArray();
    int[] s2 = new HashSet<int>(nums2).ToArray();
    foreach (int x in s1)
      if (Array.IndexOf(s2,x) < 0 ) ans[0].Add(x);
    foreach (int x in s2)
      if (Array.IndexOf(s1,x) < 0 ) ans[1].Add(x);
    return ans;
    }
}
// 2208. Minimum Operations to Halve Array Sum
// use max heap priority queue, slow
/*Solution: Greedy + PriorityQueue/Max Heap
Always half the largest number, put all the numbers onto a max heap (priority queue), extract the largest one, and put reduced number back.

Time complexity: O(nlogn)
Space complexity: O(n)*/
public class Solution {
    public int HalveArray(int[] nums) {
        double sum = 0.0 ; 
    PriorityQueue<double,double> q = new PriorityQueue<double,double>(Comparer<double>.Create((x,y) => y.CompareTo(x)));
    foreach (int num in nums) {
        q.Enqueue(num, num);
        sum += num;
    }
    int ans = 0;
    for (double cur = sum; cur > (sum / 2) ; ++ans) {    
      double num = q.Peek(); q.Dequeue();
      cur -= num / 2;
      q.Enqueue(num / 2, num / 2);
    }
    return ans;
    }
}

public class Solution {
    public int HalveArray(int[] nums) {
        double sum = 0.0 ; 
    PriorityQueue<double,double> q = new PriorityQueue<double,double>();
    foreach (int num in nums) {
        q.Enqueue(num, -num);//max heap
        sum += num;
    }
    int ans = 0;
    for (double cur = sum; cur > (sum / 2) ; ++ans) {    
      double num = q.Peek(); q.Dequeue();//get the max value of pq
      cur -= num / 2;
      q.Enqueue(num / 2, -(num / 2));//push half of max back to pq
    }
    return ans;
    }
}

// 2206. Divide Array Into Equal Pairs
// HashSet 
/*Solution: Hashtable
Each number has to appear even numbers in order to be paired. 
Count the frequency of each number, return true if all of them are even numbers,
 return false otherwise.

Time complexity: O(n)
Space complexity: O(n)

*/
public class Solution {
    public bool DivideArray(int[] nums) {
        HashSet<int> seen = new HashSet<int>();
        foreach (int num in nums) {
            if (!seen.Add(num)) {
                seen.Remove(num);
            }
        }
        return seen.Count == 0;
    }
}
// Dictionary
public class Solution {
    public bool DivideArray(int[] nums) {
        Dictionary<int, int> m = new Dictionary<int, int>();
    foreach (int num in nums)
      m[num] = m.GetValueOrDefault(num,0)+1;
     foreach (var kv in m) {
            if (kv.Value % 2 != 0)
                return false;
        }
        return true;
    }
}

// 2197. Replace Non-Coprime Numbers in Array
/*Solution: Stack
“””It can be shown that replacing adjacent non-coprime numbers in 
any arbitrary order will lead to the same result.”””

So that we can do it in one pass from left to right using a stack/vector.

Push the current number onto stack, and merge top two if they are not co-prime.

Time complexity: O(nlogm)
Space complexity: O(n)

*/
public class Solution {
    public IList<int> ReplaceNonCoprimes(int[] nums) {
        IList<int> ans = new List<int>();
    foreach (int x in nums) {
      ans.Add(x);
      while (ans.Count > 1) {
        int n1 = ans[ans.Count - 1]; 
        int n2 = ans[ans.Count - 2]; 
        int d = gcd(n1, n2);
        if (d == 1) break;
        ans.RemoveAt(ans.Count - 1);
        ans.RemoveAt(ans.Count - 1);
        ans.Add(n1 / d * n2);
      }
    }
    return ans;
    }
    
    private int gcd(int a, int b) {
        return b > 0 ? gcd(b, a % b) : a;
    }
}
// 2192. All Ancestors of a Node in a Directed Acyclic Graph
/*Solution: DFS
For each source node S, add it to all its reachable nodes by 
traversing the entire graph.
In one pass, only traverse each child node at most once.

Time complexity: O(VE)
Space complexity: (V+E)*/
public class Solution {
    public IList<IList<int>> GetAncestors(int n, int[][] edges) {
        IList<IList<int>> ans = new List<IList<int>>();
    List<List<int>> g = new List<List<int>>();
        for(int i = 0; i < n; i++){
            g.Add(new List<int>());
            ans.Add(new List<int>());
        }
    foreach (int[] e in edges)
      g[e[0]].Add(e[1]);    
    
    void dfs(int s, int u)  {
      foreach (int v in g[u]) {
        if (ans[v].Count == 0 || ans[v][ans[v].Count - 1] != s) {
          ans[v].Add(s);
          dfs(s, v);
        }
      }
    };
    
    for (int i = 0; i < n; ++i)
      dfs(i, i);  
    return ans;
    }
}

// 2190. Most Frequent Number Following Key In an Array
//Linq
public class Solution {
    public int MostFrequent(int[] nums, int key) {
        return nums
            .Skip(1)
            .Zip(nums, (a, b) => b == key ? a : -1)
            .Where(x => x > 0)
            .GroupBy(x => x)
            .MaxBy(g => g.Count())
            .Key;
    }
}
// Dictionary
/*Solution: Hashtable
Time complexity: O(n)
Space complexity: O(n)

*/
public class Solution {
    public int MostFrequent(int[] nums, int key) {
        int n = nums.Length;
    Dictionary<int, int> m = new Dictionary<int, int>();
    int count = 0;
    int ans = 0;
    for (int i = 1; i < n; ++i) {
        
      if (nums[i - 1] == key ) {
          m[nums[i]] = m.GetValueOrDefault(nums[i],0)+1;
          if(m[nums[i]] > count){
              count = m[nums[i]];
            ans = nums[i];
          }
      }
    }
    return ans;
    }
}
// Dictionary
public class Solution {
    public int MostFrequent(int[] nums, int key) {
        int n = nums.Length;
    Dictionary<int, int> m = new Dictionary<int, int>();
    int count = 0;
    int ans = 0;
    for (int i = 1; i < n; ++i) {
        
      if (nums[i - 1] == key ) {
          if (!m.ContainsKey(nums[i])) m[nums[i]] = 0;
                m[nums[i]]++;
          if(m[nums[i]] > count){
              count = m[nums[i]];
            ans = nums[i];
          }
      }
    }
    return ans;
    }
}
// 2196. Create Binary Tree From Descriptions
/*Solution: Hashtable + Recursion
Use one hashtable to track the children of each node.
Use another hashtable to track the parent of each node.
Find the root who doesn’t have parent.
Build the tree recursively from root.
Time complexity: O(n)
Space complexity: O(n)

*/
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public TreeNode CreateBinaryTree(int[][] descriptions) {
        HashSet<int> hasParent = new HashSet<int>();
    Dictionary<int, KeyValuePair<int,int>> children = new Dictionary<int, KeyValuePair<int,int>>();
    foreach (int[] d in descriptions) {  
         
      hasParent.Add(d[1]);
       if(!children.ContainsKey(d[0])) {
                children[d[0]] = new KeyValuePair<int, int>(0,0);
            }
       
      if (d[2] > 0 ){
          children[d[0]] = new KeyValuePair<int, int>(d[1],children[d[0]].Value); 
      }   
        else {
            children[d[0]] = new KeyValuePair<int, int>(children[d[0]].Key,d[1]);
        }
        
    }
    int root = -1;
    foreach (int[] d in descriptions){
         
      if (!hasParent.Contains(d[0])) {
          root = d[0];
      }
    }
       
    TreeNode build(int cur) {      
      if (cur == 0 ) return null;
        TreeNode newNode = new TreeNode(cur);
        if(children.ContainsKey(cur)){
            
            if(children[cur].Key != 0){
                newNode.left = build(children[cur].Key);
            }
            if(children[cur].Value != 0){
                newNode.right = build(children[cur].Value);  
            }
        }        
      return newNode;
    };  
    return build(root);

    }
}
 /*Explanation
Iterate descriptions,
for each [p, c, l] of [parent, child, isLeft]

Create Treenode with value p and c,
and store them in a hash map with the value as key,
so that we can access the TreeNode easily.

Based on the value isLeft,
we assign Treenode(parent).left = Treenode(child)
or Treenode(parent).right = Treenode(child).

Finall we find the root of the tree, and return its TreeNode.*/
public class Solution {
    public TreeNode CreateBinaryTree(int[][] descriptions) { 
         TreeNode root = null;
        Dictionary<int,TreeNode> map = new Dictionary<int,TreeNode>();
        foreach(int [] des in descriptions) {
            map[des[1]] = new TreeNode(des[1]);
        }
        foreach(int [] des in descriptions) {
            if(!map.ContainsKey(des[0])) {
                root = new TreeNode(des[0]);
                map[des[0]] = root;
            }
            TreeNode parent = map[des[0]];
            TreeNode cur =  map[des[1]];
            if(des[2] == 1){
                parent.left = cur;
            }else{
                parent.right = cur;
            }
        }
        return root;
    }
}
// 2195. Append K Integers With Minimal Sum
/*Solution: Greedy + Math, fill the gap
Sort all the numbers and remove duplicated ones, 
and fill the gap between two neighboring numbers.
e.g. [15, 3, 8, 8] => sorted = [3, 8, 15]
fill 0->3, 1,2, sum = ((0 + 1) + (3-1)) * (3-0-1) / 2 = 3
fill 3->8, 4, 5, 6, 7, sum = ((3 + 1) + (8-1)) * (8-3-1) / 2 = 22
fill 8->15, 9, 10, …, 14, …
fill 15->inf, 16, 17, …

Time complexity: O(nlogn)
Space complexity: O(1)*/
public class Solution {
    public long MinimalKSum(int[] nums, int k) {
        Array.Sort(nums);
    long ans = 0;
    long p = 0;
    foreach (int c in nums) {
      if (c == p) continue;
      long n = Math.Min((long)k, (long)(c - p - 1));
      ans += (p + 1 + p + n) * n / 2; 
      k -= (int)n;
      p = c;
    }
    ans += (p + 1 + p + k) * k / 2;
    return ans;
    }
}
// 2194. Cells in a Range on an Excel Sheet
/*Solution: Brute Force
Time complexity: O((row2 – row1 + 1) * (col2 – col1 + 1))
Space complexity: O(1)*/
public class Solution {
    public IList<string> CellsInRange(string s) {
        IList<string> ans = new List<string>();
    for (char c = s[0]; c <= s[3]; ++c)
      for (char r = s[1]; r <= s[4]; ++r)        
        ans.Add(c.ToString()+r.ToString());
    return ans;
    }
}
// 2178. Maximum Split of Positive Even Integers
/*Solution: Greedy
The get the maximum number of elements, we must use the smallest numbers.

[2, 4, 6, …, 2k, x], where x > 2k
let s = 2 + 4 + … + 2k, x = num – s
since num is odd and s is also odd, so thus x = num – s.

Time complexity: O(sqrt(num)) for constructing outputs.
Space complexity: O(1)*/
public class Solution {
    public IList<long> MaximumEvenSplit(long finalSum) {
        if ((finalSum & 1) > 0) return new List<long>{};
    IList<long> ans = new List<long>();
    long s = 0;
    for (long i = 2; s + i <= finalSum; s += i, i += 2)
      ans.Add(i);
    ans[ans.Count - 1] += (finalSum - s);
    return ans;
    }
}
// 2177. Find Three Consecutive Integers That Sum to a Given Number
/*Solution: Math
(x / 3 – 1) + (x / 3) + (x / 3 + 1) == 3x == num, num must be divisible by 3.

Time complexity: O(1)
Space complexity: O(1)*/
public class Solution {
    public long[] SumOfThree(long num) {
        if ((num % 3) > 0) return new long[]{};
    return new long[]{num / 3 - 1, num / 3, num / 3 + 1};    
    }
}

// 2176. Count Equal and Divisible Pairs in an Array
/*Solution: Brute Force
Time complexity: O(n2)
Space complexity: O(1)

*/
public class Solution {
    public int CountPairs(int[] nums, int k) {
        int n = nums.Length;
    int ans = 0;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        ans += Convert.ToInt32((nums[i] == nums[j]) && (i * j % k == 0));
    return ans;
    }
}
// 2180. Count Integers With Even Digit Sum
/*Solution: Brute Force
Use std::to_string to convert an integer to string.

Time complexity: O(nlgn)
Space complexity: O(lgn)*/
public class Solution {
    public int CountEven(int num) {
        int ans = 0;
    for (int i = 1; i <= num; ++i) {
      int sum = 0;
      foreach (char c in i.ToString())
        sum += (c - '0');
      if (sum % 2 == 0) ++ans;
    }
    return ans;
    }
}
// 2181. Merge Nodes in Between Zeros
/*Solution: List
Skip the first zero, replace every zero node with the sum of values of its previous nodes.

Time complexity: O(n)
Space complexity: O(1)

*/
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
public class Solution {
    public ListNode MergeNodes(ListNode head) {
        ListNode dummy = new ListNode(0);    
    head = head.next;
    for (ListNode prev = dummy; head != null; head = head.next) {
      int sum = 0;
      while (head.val != 0) {
        sum += head.val;
        head = head.next;
      }
      prev.next = head;
      head.val = sum;
      prev = head;      
    }
    return dummy.next;
    }
}
// 2182. Construct String With Repeat Limit
// Greedy TLE 
/*Solution: Greedy
Adding one letter at a time, find the largest one that can be used.

Time complexity: O(26*n)
Space complexity: O(1)*/
public class Solution {
    public string RepeatLimitedString(string s, int repeatLimit) {
     int[] m = new int[26];
    foreach (char c in s) ++m[c - 'a'];
    string ans = String.Empty;
    int count = 0;
    int last = -1;
    bool found = true; 
    while (found) {
      found = false;      
      for (int i = 25; i >= 0 && !found; --i)
        if (m[i] > 0 && (count < repeatLimit || last != i)) {
          ans += Convert.ToChar('a' + i);
          ++count;if( last!=i ) count = 1;
          --m[i];
          last = i;
          found = true;          
        }
    }
    return ans;

    }
}
// Greedy
public class Solution {
    public string RepeatLimitedString(string s, int repeatLimit) {
        int[] cnt = new int[26];
        int n = s.Length;
        for (int i = 0; i < n; ++i) {
            ++cnt[s[i] - 'a'];
        } 
        StringBuilder res = new StringBuilder();
        for (char c = 'z'; c >= 'a'; --c) {
            while (cnt[c - 'a'] > 0) {
                for (int i = 0; i < repeatLimit && cnt[c - 'a'] > 0; ++i) {
                    --cnt[c - 'a'];
                    res.Append(c);
                }
                if (cnt[c - 'a'] == 0) break;
                char take = '0';
                for (char d = 'a'; d <= 'z'; ++d) {
                    if (d != c && cnt[d - 'a'] > 0) {
                        take = d;
                    }
                }
                if (take == '0') break;
                --cnt[take - 'a'];
                res.Append(take);
            }
        }
        return Convert.ToString(res);
    }
}
// 2183. Count Array Pairs Divisible by K
/*Solution: Math
a * b % k == 0 <=> gcd(a, k) * gcd(b, k) == 0

Use a counter of gcd(x, k) so far to compute the number of pairs.

Time complexity: O(n*f), where f is the number of gcds, f <= 128 for x <= 1e5
Space complexity: O(f)*/
public class Solution {
    public long CountPairs(int[] nums, int k) {
         Dictionary<int, int> m = new Dictionary<int, int>();
    long ans = 0;
    foreach (int x in nums) {
      long d1 = gcd(x, k);
      foreach ( var (d2, c) in m)
        if (d1 * d2 % k == 0) ans += c;
      m[(int)d1] = m.GetValueOrDefault((int)d1,0)+1;
    }
    return ans;
    }
     private int gcd(int a, int b) {
        return b > 0 ? gcd(b, a % b) : a;
    }
}
// 2188. Minimum Time to Finish the Race
/*Solution: DP
Observation: since ri >= 2, we must change tire within 20 laps, otherwise it will be slower.

pre-compute the time to finish k laps using each type of tire (k < 20), find min for each lap.

dp[i] = best[i], i < 20,
dp[i] = min{dp[i – j] + changeTime + best[j]}, i > 20

Time complexity: O(n)
Space complexity: O(n) -> O(20)*/
public class Solution {
    public int MinimumFinishTime(int[][] tires, int changeTime, int numLaps) {
         int kMax = 20;
    long[] best = new long[kMax];Array.Fill(best, int.MaxValue);
    foreach(int[] t in tires)
      for (long i = 1, l = t[0], s = t[0]; i < kMax && l < t[0] + changeTime; ++i, l *= t[1], s += l)
        best[i] = Math.Min(best[i], s);
    long[] dp = new long[numLaps + 1]; Array.Fill(dp,int.MaxValue);    
    for (int i = 1; i <= numLaps; ++i)
      for (int j = 1; j <= Math.Min(i, kMax - 1); ++j)
        dp[i] = Math.Min(dp[i], (best[j] + Convert.ToInt64(i > j) * (changeTime + dp[i - j])));
    return (int)dp[numLaps];
    }
}
// 2187. Minimum Time to Complete Trips
/*Solution: Binary Search
Find the smallest t s.t. trips >= totalTrips.

Time complexity: O(nlogm), where m ~= 1e15
Space complexity: O(1)*/
public class Solution {
    public long MinimumTime(int[] time, int totalTrips) {
        long l = 1;
    long r = Convert.ToInt64(1e15);
    while (l < r) {
      long m = l + (r - l) / 2;
      long trips = 0;
      foreach (int t in time) {        
        trips += m / t;
        if (trips >= totalTrips) break;
      }
      if (trips >= totalTrips)
        r = m;
      else
        l = m + 1;
    }
    return l;
    }
}
// 2186. Minimum Number of Steps to Make Two Strings Anagram II
/*Solution: Hashtable
Record the frequency difference of each letter.

Ans = sum(diff)

Time complexity: O(m + n)
Space complexity: O(26)*/
public class Solution {
    public int MinSteps(string s, string t) {
         int[] diff = new int[26];
    foreach (char c in s) ++diff[c - 'a'];
    foreach (char c in t) --diff[c - 'a'];
    int ans = 0;
    foreach (int d in diff)
      ans += Math.Abs(d);
    return ans;
    }
}
// 2185. Counting Words With a Given Prefix
/*Solution: Straight forward
We can use std::count_if and std::string::find.

Time complexity: O(n*l)
Space complexity: O(1)*/
public class Solution {
    public int PrefixCount(string[] words, string pref) {
        int count = 0;
        foreach (string w in words){
            if (w.IndexOf(pref) == 0) count++; 
        }
        return count;
    }
}
// 2169. Count Operations to Obtain Zero
/*Solution 2: Simualtion + Math
For the case of 100, 3
100 – 3 = 97
97 – 3 = 94
…
4 – 3 = 1
Swap
3 – 1 = 2
2 – 1 = 1
1 – 1 = 0
It takes 36 steps.

We can do 100 / 3 to skip 33 steps
100 %= 3 = 1
3 / 1 = 3 to skip 3 steps
3 %= 1 = 0
total is 33 + 3 = 36.

Time complexity: O(logn) ?
Space complexity: O(1)*/
public class Solution {
    public int CountOperations(int num1, int num2) {
         int ans = 0;
    while (num1 > 0 && num2 > 0) {
      if (num1 < num2) {
          int tmp = num1;
          num1 = num2;
          num2 = tmp;
      }
      ans += num1 / num2;
      num1 %= num2;      
    }
    return ans;
    }
}

// 2147. Number of Ways to Divide a Long Corridor
/*Solution: Combination
If the 2k-th seat is positioned at j, and the 2k+1-th seat is at i. 
There are (i – j) ways to split between these two groups.

ans = prod{ik – jk}

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int NumberOfWays(string corridor) {
        double kMod = 1e9 + 7;
    long ans = 1;
    long k = 0;
    for (int i = 0, j = 0; i < corridor.Length; ++i) {
      if (corridor[i] != 'S') continue;
      if (++k > 2 && (k & 1) > 0)
        ans = ans * (i - j) % Convert.ToInt64(kMod);
      j = i;
    }
    return (k >= 2 && k % 2 == 0) ? Convert.ToInt32(ans) : 0;
    }
}

// 2146. K Highest Ranked Items Within a Price Range
/*Solution: BFS + Sorting
Use BFS to collect reachable cells and sort afterwards.

Time complexity: O(mn + KlogK) where K = # of reachable cells.

Space complexity: O(mn)*/
public class Solution {
    public IList<IList<int>> HighestRankedKItems(int[][] grid, int[] pricing, int[] start, int k) {
        int m = grid.Length;
    int n = grid[0].Length;    
    int[] dirs = new int[5]{1, 0, -1, 0, 1};    
    int[][] seen = new int[m][];
        for(int i = 0; i < m;i++) seen[i]= new int[n];
    seen[start[0]][start[1]] = 1;
    List<List<int>> cells = new List<List<int>>();
    Queue<int[]> q = new Queue<int[]>();
    q.Enqueue(new int[]{start[0], start[1], 0});
    while (q.Count != 0) {
      var a = q.Peek(); q.Dequeue();
       //a = [y, x, d] 
        int y = a[0], x = a[1], d = a[2];
      if (grid[y][x] >= pricing[0] && grid[y][x] <= pricing[1])
          cells.Add(new List<int>{d, grid[y][x], y, x});
      for (int i = 0; i < 4; ++i) {
        int tx = x + dirs[i];
        int ty = y + dirs[i + 1];
        if (tx < 0 || tx >= n || ty < 0 || ty >= m 
            || grid[ty][tx] == 0 || seen[ty][tx]++ > 0) continue;
        q.Enqueue(new int[]{ty, tx, d + 1});
      }
    }
    //sort(begin(cells), end(cells), less<vector<int>>());
        cells.Sort((a, b) => a[0] == b[0] ? a[1] == b[1] ? a[2] == b[2] ? a[3] - b[3] : a[2] - b[2] : a[1] - b[1] : a[0] - b[0]);//works!
    IList<IList<int>> ans = new List<IList<int>>();
    for (int i = 0; i < Math.Min(k, (int)(cells.Count)); ++i)
      ans.Add(new List<int>{cells[i][2], cells[i][3]});
    return ans;
    }
}
// BFS with queue
public class Solution {
    public IList<IList<int>> HighestRankedKItems(int[][] grid, int[] pricing, int[] start, int k) {
        var rows = grid.Length;
        var cols = grid[0].Length;
        
        var res = new List<IList<int>>();
        var queue = new Queue<(int row, int col)>();
        queue.Enqueue((start[0], start[1]));
        
        while(queue.Count > 0)
        {
            var list = new List<(int row, int col, int price)>();
            var qc = queue.Count;
            for(int i=0; i<qc; i++)
            {
                var e = queue.Dequeue();

                var price = grid[e.row][e.col];
                if(price == 0)
                    continue;
                if(price > 1 && price >= pricing[0] && price <= pricing[1])
                    list.Add((e.row, e.col, price));
                grid[e.row][e.col] = 0;

                if(e.row + 1 < rows && grid[e.row + 1][e.col] != 0)
                    queue.Enqueue((e.row + 1, e.col));
                if(e.row - 1 >= 0 && grid[e.row - 1][e.col] != 0)
                    queue.Enqueue((e.row - 1, e.col));
                if(e.col + 1 < cols && grid[e.row][e.col + 1] != 0)
                    queue.Enqueue((e.row, e.col + 1));
                if(e.col - 1 >= 0 && grid[e.row][e.col - 1] != 0)
                    queue.Enqueue((e.row, e.col - 1));
            }
            var sorted = list.OrderBy(e => e.price).ThenBy(e => e.row).ThenBy(e => e.col);
            foreach(var e in sorted)
                if(res.Count < k)
                    res.Add(new int[] {e.row, e.col});
                else
                    break;
            
            if(res.Count == k)
                break;
        }
        return res;
    }
}
// 2145. Count the Hidden Sequences
/*Solution: Math
Find the min and max of the cumulative sum of the differences.

Ans = max(0, upper – lower – (hi – lo) + 1)

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int NumberOfArrays(int[] differences, int lower, int upper) {
        long s = 0;
    long hi = 0;
    long lo = 0;
    foreach (int d in differences) {
      s += d;
      hi = Math.Max(hi, s);
      lo = Math.Min(lo, s);
    }
    return (int)Math.Max(0, upper - lower - (hi - lo) + 1);
    }
}
// 2144. Minimum Cost of Buying Candies With Discount
/*Solution: Greedy
Sort candies in descending order. 
Buy 1st, 2nd, take 3rd, buy 4th, 5th take 6th, …

Time complexity: O(nlogn)
Space complexity: O(1)*/
public class Solution {
    public int MinimumCost(int[] cost) {
        Array.Sort(cost, (a, b) => b - a);
        List<int> c = cost.ToList();
    while ((c.Count % 3 ) != 0) c.Add(0);
    int ans = 0;
    for (int i = 0; i < c.Count; i += 3)
      ans += c[i] + c[i + 1];
    return ans;
       
    }
}
/*Explanation
For the max value, we have to pay for it.
For the second max value, we still have to pay for it.
For the third max value, we can get it free one as bonus.
And continuely do this for the rest.

The the core of problem, is need to sort the input.
All A[i] with i % 3 == n % 3, we can get it for free.


Complexity
Time O(sort)
Space O(sort)*/
public class Solution {
    public int MinimumCost(int[] cost) {
        Array.Sort(cost);
        int res = 0, n = cost.Length;
        for (int i = 0; i < n; ++i)
            if (i % 3 != n % 3)
                res += cost[i];
        return res;
    }
}
//Linq
public class Solution {
    public int MinimumCost(int[] cost) => cost.OrderByDescending(x => x)
		       .Where((number, index) => index%3 != 2)
			   .Sum();
}
/*Intuition
maximize the cost of free candies
sort the array in descending order
skip candies at index 2, 5, 8... by using (i+1) % 3 == 0

Complexity

Time: O(nlogn) where n = cost.Length
Space: O(1)*/
public class Solution {
    public int MinimumCost(int[] cost) {
        
        Array.Sort(cost, (a,b) => b - a);
        int res = 0;
        
        for(int i = 0; i < cost.Length; i++)
        {
            if((i + 1) % 3 != 0)
                res += cost[i];
        }
        
        return res;
       
    }
}

// 2141. Maximum Running Time of N Computers
/*Solution: Binary Search
Find the smallest L that we can not run, ans = L – 1.

For a guessing m, we check the total battery powers T = sum(min(m, batteries[i])),
 if T >= m * n, it means there is a way (doesn’t need to figure out how) 
 to run n computers for m minutes by fully unitize those batteries.

Proof: If T >= m*n holds, there are two cases:

There are only n batteries, can not swap, but each of them has power >= m.
At least one of the batteries have power less than m, but there are more than n batteries and total power is sufficient, we can swap them with others.
Time complexity: O(Slogn) where S = sum(batteries)
Space complexity: O(1)

*/
public class Solution {
    public long MaxRunTime(int n, int[] batteries) {
        long l = 0;
    long r = 1; Array.ForEach(batteries, i => r+=i);
    while (l < r) {
      long m = l + (r - l) / 2;
      long t = 0;
      foreach (int b in batteries)
        t += Math.Min(m, b);
      if (m * n > t) // smallest m that does not fit.
        r = m;
      else
        l = m + 1;
    }
    return l - 1; // greatest m that fits.
    }
}

// 2154. Keep Multiplying Found Values by Two
/*Solution: Hashset
Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int FindFinalValue(int[] nums, int original) {
        HashSet<int> s = new HashSet<int>(nums);
    while (true) {
      if (!s.Contains(original)) break;
      original *= 2;
    }
    return original;
    }
}

// 2151. Maximum Good People Based on Statements
/*Solution: Combination / Bitmask
Enumerate all subsets of n people and assume they are good people. 
Check whether their statements have any conflicts. 
We can ignore the statements from bad people since those can be either true or 
false and does not affect our checks.

Time complexity: O(n22n)
Space complexity: O(1)

*/
public class Solution {
    public int MaximumGood(int[][] statements) {
        int n = statements.Length;
    bool valid(int s) {
      for (int i = 0; i < n; ++i) {
        if (!Convert.ToBoolean(s >> i & 1)) continue;
        for (int j = 0; j < n; ++j) {
            bool good = Convert.ToBoolean( s >> j & 1);
          if ((good && statements[i][j] == 0) || (!good && statements[i][j] == 1))
            return false;
        }
      }
      return true;
    };
    //int CountOnes(int mask) => Convert.ToString(mask, 2).Count('1'.Equals);
    int ans = 0;
    for (int s = 1; s < 1 << n; ++s)
      if (valid(s)) ans = Math.Max(ans, Convert.ToString(s, 2).Count(f => f == '1'));//Convert.ToString(s, 2).Count(f => f == '1')
    return ans;
    }
}
public class Solution {
    public int MaximumGood(int[][] statements) {
        int n = statements.Length;
    bool valid(int s) {
      for (int i = 0; i < n; ++i) {
        if (!Convert.ToBoolean(s >> i & 1)) continue;
        for (int j = 0; j < n; ++j) {
            bool good = Convert.ToBoolean( s >> j & 1);
          if ((good && statements[i][j] == 0) || (!good && statements[i][j] == 1))
            return false;
        }
      }
      return true;
    };
    //int CountOnes(int mask) => Convert.ToString(mask, 2).Count('1'.Equals);
    int ans = 0;
    for (int s = 1; s < 1 << n; ++s)
      if (valid(s)) ans = Math.Max(ans, Convert.ToString(s, 2).Count('1'.Equals)); //Convert.ToString(s, 2).Count('1'.Equals)
    return ans;
    }
}
public class Solution {
    public int MaximumGood(int[][] statements) {
        int n = statements.Length;
    bool valid(int s) {
      for (int i = 0; i < n; ++i) {
        if (!Convert.ToBoolean(s >> i & 1)) continue;
        for (int j = 0; j < n; ++j) {
            bool good = Convert.ToBoolean( s >> j & 1);
          if ((good && statements[i][j] == 0) || (!good && statements[i][j] == 1))
            return false;
        }
      }
      return true;
    };
    int CountOnes(int mask) => Convert.ToString(mask, 2).Count('1'.Equals);
    int ans = 0;
    for (int s = 1; s < 1 << n; ++s)
      if (valid(s)) ans = Math.Max(ans, CountOnes(s));
    return ans;
    }
}
//Linq
public class Solution {
    public int MaximumGood(int[][] statements) {
        int[] haveOpinion = (from array in statements
                             select Enumerable.Range(0, array.Length).Sum(i => array[i] != 2 ? 1 << i : 0)).ToArray();
        
        int[] good = (from array in statements
                      select Enumerable.Range(0, array.Length).Sum(i => array[i] == 1 ? 1 << i : 0)).ToArray();
        
        return Enumerable
            .Range(0, 1 << statements.Length)
            .Where(mask => Enumerable
                        .Range(0, statements.Length)
                        .Where(i => (mask & (1 << i)) > 0)
                        .All(i => good[i] == (mask & haveOpinion[i])))
            .Max(mask => Convert.ToString(mask, 2).Count('1'.Equals));
    }
    
}
// 2150. Find All Lonely Numbers in the Array
/*Solution: Counter
Computer the frequency of each number in the array, 
for a given number x with freq = 1, check freq of (x – 1) and (x + 1), 
if both of them are zero then x is lonely.

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public IList<int> FindLonely(int[] nums) {
        Dictionary<int, int> m = new Dictionary<int, int>();
    foreach (int x in nums)
      m[x] = m.GetValueOrDefault(x,0)+1;
    IList<int> ans = new List<int>();
    foreach (var (x, c) in m)
      if (c == 1 && !m.ContainsKey(x + 1) && !m.ContainsKey(x - 1))
        ans.Add(x);
    return ans;
    }
}
// 2149. Rearrange Array Elements by Sign
/*Solution 1: Split and merge
Create two arrays to store positive and negative numbers.

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int[] RearrangeArray(int[] nums) {
        List<int> pos = new List<int>();
    List<int> neg = new List<int>();
    foreach (int x in nums)
      (x > 0 ? pos : neg).Add(x);
    List<int> ans = new List<int>();
    for (int i = 0; i < pos.Count; ++i) {
      ans.Add(pos[i]);
      ans.Add(neg[i]);
    }
    return ans.ToArray();
    }
}
/*Solution 2: Two Pointers
Use two pointers to store the next pos / neg.

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int[] RearrangeArray(int[] nums) {
        int[] ans = new int[nums.Length];    
    int pos = 0;
    int neg = 1;
    foreach (int x in nums) {
      ans[(x > 0) ? pos : neg] = x;
      if (x > 0) pos += 2; else neg += 2;
    }
    return ans;
    }
}
public class Solution {
    public int[] RearrangeArray(int[] nums) {
        int[] ans = new int[nums.Length];    
    int pos = 0;
    int neg = 1;
    foreach (int x in nums) {
        if(x>0){
                ans[pos] = x;
                pos+=2;
            }
            if(x<0){
                ans[neg] = x;
                neg += 2;
            }
    }
    return ans;
    }
}
// 2148. Count Elements With Strictly Smaller and Greater Elements
/*Solution: Min / Max elements

Find min and max of the array, count elements other than those two.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int CountElements(int[] nums) {
        int count = 0, lo = nums.Min(), hi = nums.Max();
        foreach(int n in nums){
            if(n != lo && n != hi) count++;
        }
        return count;
    }
}
// 2155. All Divisions With the Highest Score of a Binary Array
/*Solution: Precompute + Prefix Sum
Count how many ones in the array, track the prefix sum to compute score for each index in O(1).

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public IList<int> MaxScoreIndices(int[] nums) {
        int n = nums.Length;
    int t = nums.Sum(); //quicker!
    // int t =  0; Array.ForEach(nums, i => t += i); //works! slower!
    IList<int> ans = new List<int>();
    for (int i = 0, p = 0, b = 0; i <= n; ++i) {
      int s = (i - p) + (t - p);      
      if (s > b) {
        b = s; ans.Clear();
      }
      if (s == b) ans.Add(i);
      if (i != n) p += nums[i];
    }
    return ans;
    }
}

// 2156. Find Substring With Given Hash Value
/*Solution: Sliding window
hash = (((hash – (s[i+k] * pk-1) % mod + mod) * p) + (s[i] * p0)) % mod

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public string SubStrHash(string s, int power, int modulo, int k, int hashValue) {
        int n = s.Length;
    long p = 1, cur = 0;
    for (int i = 1; i < k; ++i)
      p = (p * power) % modulo;
    int ans = 0; 
    for (int i = n - 1; i >= 0; --i) {
      if (i + k < n) 
        cur = ((cur - (s[i + k] - 'a' + 1) * p) % modulo + modulo) % modulo;
      cur = (cur * power + (s[i] - 'a' + 1)) % modulo;      
      if (i + k <= n && cur == hashValue) 
        ans = i;
    }    
    return s.Substring(ans, k);    
    }
}
/*Sliding Window + Rolling Hash
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
Space O(1)*/
public class Solution {
    public string SubStrHash(string s, int p, int m, int k, int hashValue) {
        long cur = 0, pk = 1;
        int res = 0, n = s.Length;
        for (int i = n - 1; i >= 0; --i) {
            cur = (cur * p + s[i] - 'a' + 1) % m;
            if (i + k >= n)
                pk = pk * p % m;
            else
                cur = (cur - (s[i + k] - 'a' + 1) * pk % m + m) % m;
            if (cur == hashValue)
                res = i;
        }
        return s.Substring(res, k);
    }
}

// 2157. Groups of Strings
/*Solution: Bitmask + DFS
Use a bitmask to represent a string. Use dfs to find connected components.

Time complexity: O(n*262)
Space complexity: O(n)

*/
public class Solution {
    public int[] GroupStrings(string[] words) {
        Dictionary<int, int> m = new Dictionary<int, int>();
    
        foreach(string w in words)
        {
            int x = 0;
            foreach(char ch in w)
                x = x | 1 << (ch-'a');
            
            m[x] = m.GetValueOrDefault(x,0) +1;
       
        }
        
    int dfs(int mask) {
      //int it = m.ContainsKey(mask);
      if (!m.ContainsKey(mask)) return 0;
      int ans = m[mask];      
      m.Remove(mask);
      for (int i = 0; i < 26; ++i) {        
        ans += dfs(mask ^ (1 << i));
        for (int j = i + 1; j < 26; ++j)
          if ((mask >> i & 1) != (mask >> j & 1))
            ans += dfs(mask ^ (1 << i) ^ (1 << j));
      }
      return ans;
    };
    int size = 0;
    int groups = 0;
    while (m.Count != 0) {
      size = Math.Max(size, dfs(m.ElementAt(0).Key));
      ++groups;
    }
    return new int[] {groups, size};
    }
}
/*some common bitmask operations:

take the bit of the ith(from right) digit:
		bit = (mask >> i) & 1;
set the ith digit to 1:
		mask = mask | (1 << i);
set the ith digit to 0:
		mask = mask & (~(1 << i));*/
public class Solution {
    public int[] GroupStrings(string[] words) {
        Dictionary<int, int> m = new Dictionary<int, int>();
    
    for(int i = 0; i < words.Length; i++)
        {
            int mask = 0;
            foreach(char ch in words[i])
                mask = mask | 1 << (ch-'a');
            
            if(m.ContainsKey(mask))
                m[mask]++;
            else
                m.Add(mask, 1);            
        }
        
    int dfs(int mask) {
      //int it = m.ContainsKey(mask);
      if (!m.ContainsKey(mask)) return 0;
      int ans = m[mask];      
      m.Remove(mask);
      for (int i = 0; i < 26; ++i) {        
        ans += dfs(mask ^ (1 << i));
        for (int j = i + 1; j < 26; ++j)
          if ((mask >> i & 1) != (mask >> j & 1))
            ans += dfs(mask ^ (1 << i) ^ (1 << j));
      }
      return ans;
    };
    int size = 0;
    int groups = 0;
    while (m.Count != 0) {
      size = Math.Max(size, dfs(m.ElementAt(0).Key));
      ++groups;
    }
    return new int[] {groups, size};
    }
}
// bitmask and dfs
public class Solution {
    public int[] GroupStrings(string[] words) {
         Dictionary<int,int> dict = new Dictionary<int,int>();
        for(int i = 0; i < words.Length; i++)
        {
            int mask = 0;
            foreach(char ch in words[i])
                mask = mask | 1<< (ch-'a');
            
            if(dict.ContainsKey(mask))
                dict[mask]++;
            else
                dict.Add(mask, 1);            
        }
        int groups = 0, largest = 0;
        while(dict.Count > 0)
        {
            int mask = dict.Keys.First();
            int size = dfs(mask, dict);
            groups += size > 0 ? 1 : 0;
            largest = Math.Max(largest, size);
        }
        return new int[2] {groups, largest};
    }
    
    private int dfs(int mask, Dictionary<int,int> dict)
    {
        int res = 0;
        if(dict.ContainsKey(mask))
        {
            res += dict[mask];
            dict.Remove(mask);
            for(int i = 0; i < 26; i++)
            {
                res += dfs(mask ^ (1<<i), dict);
                for (int j = i + 1; j < 26; ++j)
                    if (((mask >> i) & 1) != ((mask >> j) & 1))
                        res += dfs(mask ^ (1 << i) ^ (1 << j), dict);
            }
        }
        return res;
    }
}

// 2170. Minimum Operations to Make the Array Alternating
/*Solution: Greedy
Count and sort the frequency of numbers at odd and even positions.

Case 1: top frequency numbers are different, change the rest of numbers to them respectively.
Case 2: top frequency numbers are the same, compare top 1 odd + top 2 even vs top 2 even + top 1 odd.

Time complexity: O(nlogn)
Space complexity: O(n)*/
public class Solution {
    public int MinimumOperations(int[] nums) {
        int n = nums.Length;
    if (n == 1) return 0;
   Dictionary<int, int> odd = new Dictionary<int, int>(), even = new Dictionary<int, int>();
    for (int i = 0; i < n; ++i)
        if((i & 1 )> 0 ){odd[nums[i]] = odd.GetValueOrDefault(nums[i],0)+1;}
        else{even[nums[i]] = even.GetValueOrDefault(nums[i],0)+1;}
     // (( odd : even)[nums[i]]) = ;
    List<KeyValuePair<int, int>> o = new List<KeyValuePair<int, int>>(), e = new List<KeyValuePair<int, int>>();
    foreach (var (k, v) in odd)
      o.Add( new KeyValuePair<int, int>(v, k));
    foreach (var (k, v) in even)
      e.Add( new KeyValuePair<int, int>(v, k));
      //  o.Sort((x, y) => y.Key.CompareTo(x.Key)); Works too!
       // e.Sort((x, y) => y.Key.CompareTo(x.Key));
    o = o.OrderBy(x => - x.Key).ToList();//Works!
    e = e.OrderBy(x => - x.Key).ToList();   
    if (o[0].Value != e[0].Value) 
      return n - o[0].Key - e[0].Key;    
    int mo = o[0].Key + (e.Count > 1 ? e[1].Key : 0);
    int me = e[0].Key + (o.Count > 1 ? o[1].Key : 0);
    return n - Math.Max(mo, me);
    }
}
//Linq
public class Solution {
    public int MinimumOperations(int[] nums) {
        var atEvenIndexes = nums.Where((_, i) => i % 2 == 0);
        var atOddIndexes = nums.Where((_, i) => i % 2 == 1);
        
        var top2ForOdd = Top2(atOddIndexes).ToArray();
        
        return Top2(atEvenIndexes)
                .Select(evenCandidate => 
                        (evenCandidate, oddCandidate: top2ForOdd.Except(new [] { evenCandidate }).FirstOrDefault(-1)))
                .Min(p => atEvenIndexes.Count(x => x != p.evenCandidate) + atOddIndexes.Count(x => x != p.oddCandidate));
        
        static IEnumerable<int> Top2(IEnumerable<int> nums) => nums
            .GroupBy(x => x)
            .OrderByDescending(g => g.Count())
            .Take(2)
            .Select(g => g.Key);
    }
}
// 2140. Solving Questions With Brainpower
// 2139. Minimum Moves to Reach Target Score
/*Solution: Reverse + Greedy
If num is odd, decrement it by 1. Divide num by 2 until maxdoubles times. Apply decrementing until 1 reached.

ex1: 19 (dec)-> 18 (div1)-> 9 (dec) -> 8 (div2)-> 4 (dec)-> 3 (dec)-> 2 (dec)-> 1

Time complexity: O(logn)
Space complexity: O(1)*/
public class Solution {
    public int MinMoves(int target, int maxDoubles) {
        int ans = 0;
    while (maxDoubles-- > 0 && target != 1) {
      if ((target & 1) > 0) 
      {--target; ++ans;}
      ++ans;
      target >>= 1;      
    }
    ans += (target - 1);
    return ans;
    }
}
/*Intuition
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
Space O(1)*/
public class Solution {
    public int MinMoves(int target, int maxDoubles) {
        int res = 0;
        while (target > 1 && maxDoubles > 0) {   
            res += 1 + target % 2;
            maxDoubles--;
            target >>= 1;
        }
        return target - 1 + res;
    }
}
// 2138. Divide a String Into Groups of Size k
/*Solution: Pre-fill
Time complexity: O(n)
Space complexity: O(k)*/
public class Solution {
    public string[] DivideString(string s, int k, char fill) {
        if (s.Length % k > 0){
        //s += string.Concat(Enumerable.Repeat(fill, k - s.Length % k));//works!
           s += new string(fill, k - s.Length % k);//works!
        }
            
    List<string> ans = new List<string>();
    for (int i = 0; i < s.Length; i += k)
      ans.Add(s.Substring(i, k));
    return ans.ToArray();
    }
}
public class Solution {
    public string[] DivideString(string s, int k, char fill) {
        while (s.Length % k != 0){
        s += fill;
        }
    List<string> ans = new List<string>();
    for (int i = 0; i < s.Length; i += k)
      ans.Add(s.Substring(i, k));
    return ans.ToArray();
    }
}
// With Out String Builder
public class Solution {
    public string[] DivideString(string s, int k, char fill) {
        int len = 0;
        while (s.Length % k != 0){
        s += fill;
        }
        String[] res = new String[s.Length/k];
        int i = 0;
        int index = 0;

        while (i < s.Length){
            String subStr = s.Substring(i, k);
            res[index++] = subStr;
            i = i + k;
        }
        return res;
    }
}
// 2135. Count Words Obtained After Adding a Letter
/*Solution: Bitmask w/ Hashtable
Since there is no duplicate letters in each word, we can use a bitmask to represent a word.

Step 1: For each word in startWords, we obtain its bitmask and insert it into a hashtable.
Step 2: For each word in targetWords, enumerate it’s letter and unset 1 bit (skip one letter) and see whether it’s in the hashtable or not.

E.g. for target word “abc”, its bitmask is 0…0111, and we test whether “ab” or “ac” or “bc” in the hashtable or not.

Time complexity: O(n * 26^2)
Space complexity: O(n * 26)

*/
public class Solution {
    public int WordCount(string[] startWords, string[] targetWords) {
        HashSet<int> s = new HashSet<int>();
    int getKey(string s) {      
      //return accumulate(begin(s), end(s), 0, [](int key, char c) { return key | (1 << (c - 'a')); });
         int x = 0;
            foreach(char ch in s)
               x = x | (1 << (ch-'a'));
        return x;
    };
    
    foreach (string w in startWords)
      s.Add(getKey(w));
        
    int ans = 0;
        foreach(String w in targetWords){
            int key = getKey(w);
            foreach(char c in w){
                if(s.Contains(key ^ (1 << (c - 'a')))){
                    ans++;
                    break;
                }
            }
        }
        return ans;           
    }
}
// 2134. Minimum Swaps to Group All 1's Together II
/*Solution: Sliding Window
Step 1: Count how many ones are there in the array. Assume it’s K.
Step 2: For each window of size k, count how many ones in the window, we have to swap 0s out with 1s to fill the window. ans = min(ans, k – ones).

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int MinSwaps(int[] nums) {
        int n = nums.Length;
    int k = nums.Sum();
    int ans = int.MaxValue;
    for (int i = 0, cur = 0; i < n + k; ++i) {
      if (i >= k) cur -= nums[(i - k + n) % n];
      cur += nums[i % n];      
      ans = Math.Min(ans, k - cur);
    }
    return ans;
    }
}
// 2133. Check if Every Row and Column Contains All Numbers
/*Solution: Bitset / hashtable
Time complexity: O(n2)
Space complexity: O(n)*/
public class Solution {
    public bool CheckValid(int[][] matrix) {
        int n = matrix.Length;
    for (int i = 0; i < n; ++i) {
       HashSet<int> row = new HashSet<int>();
       HashSet<int> col = new HashSet<int>();
      for (int j = 0; j < n; ++j){
          row.Add(matrix[i][j]);
            col.Add(matrix[j][i]);}
        //col[matrix[i][j]] = row[matrix[j][i]] = true;      
      if (row.Count != n || col.Count != n) 
        return false;
    }
    return true;
    }
}

// 2130. Maximum Twin Sum of a Linked List
/*Solution: Two Pointers + Reverse List
Use fast slow pointers to find the middle point and reverse the second half.

Time complexity: O(n)
Space complexity: O(1)*/
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
public class Solution {
    public int PairSum(ListNode head) {
        ListNode reverse(ListNode head, ListNode prev = null) {
      while (head != null) {
          ListNode temp = head.next;
          head.next = prev;
          prev = temp;
          
          temp = head;
          head = prev;
          prev = temp;
       // swap(head.next, prev);
       // swap(head, prev);
      }
      return prev;
    };
    
    ListNode fast = head;
    ListNode slow = head;
    while (fast != null) {
      fast = fast.next.next;
      slow = slow.next;
    }
    
    slow = reverse(slow);
    int ans = 0;
    while (slow != null) {
      ans = Math.Max(ans, head.val + slow.val);
      head = head.next;
      slow = slow.next;
    }
    return ans;
    }
}

// 2129. Capitalize the Title
/*Solution: Straight forward
Without splitting the sentence into words, we need to take care the word of length one and two.

Tips: use std::tolower, std::toupper to transform letters.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public string CapitalizeTitle(string title) {
        int n = title.Length;
        StringBuilder sb = new StringBuilder(title);
    for (int i = 0; i < title.Length; ++i) {
        
      if ((i == 0 || title[i - 1] == ' ') 
          && i + 2 < n && title[i + 1] != ' ' && title[i + 2] != ' ')
      { sb[i] = Char.Parse(title.Substring(i,1).ToUpper()); }
    
      else{ sb[i] =Char.Parse(title.Substring(i,1).ToLower());
      }
     
    }
    return sb.ToString();
    }
}
// Linq
public class Solution {
    public string CapitalizeTitle(string title)=>
         string.Join(" ",title.Split(' ')
           .Select(word => word.Length <= 2 ? word.ToLower() :
            string.Concat(char.ToUpper(word.First()) +
               string.Concat(word.Skip(1).Select(x => char.ToLower(x))))));
}

// 2126. Destroying Asteroids
/*Solution: Greedy
Sort asteroids by weight. Note, mass can be very big (105*105), for C++/Java, use long instead of int.

Time complexity: O(nlogn)
Space complexity: O(1)

*/
public class Solution {
    public bool AsteroidsDestroyed(int mass, int[] asteroids) {
        long curr = mass;
    Array.Sort(asteroids);
    foreach (int a in asteroids) {
      if (curr < a) return false;
      curr += a;
    }
    return true;
    }
}
// 2125. Number of Laser Beams in a Bank
/*Solution: Rule of product
Just need to remember the # of devices of prev non-empty row.
# of beams between two non-empty row equals to row[i] * row[j]
ans += prev * curr

Time complexity: O(m*n)
Space complexity: O(1)

*/
public class Solution {
    public int NumberOfBeams(string[] bank) {
        int ans = 0, prev = 0, count = 0;
    foreach(String s in bank) {
        count = 0;
        for (int i = 0; i < s.Length; i++) 
            if (s[i] == '1') count++;
        if (count > 0) {
            ans += prev * count;
            prev = count;
        }
    }
    return ans;
    }
}
// 2124. Check if All A's Appears Before All B's
/*Solution: Count bs
Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public bool CheckString(string s) {
        int b = 0;
    foreach (char ch in s) {
      b += Convert.ToInt32(ch == 'b');
      if (ch == 'a' && b > 0) return false;
    }
    return true;
    }
}
// 1998. GCD Sort of an Array
/*Solution: Union-Find
Let nums[j]’s target position be i. 
In order to put nums[j] to pos i by swapping. 
nums[i] and nums[j] must be in the same connected component. 
There is an edge between two numbers if they have gcd > 1.

We union two numbers if their have gcd > 1. 
However, it will be TLE if we do all pairs . 
Thus, for each number, we union it with its divisors instead.

Time complexity: O(n2) TLE -> O(sum(sqrt(nums[i]))) <= O(n*sqrt(m))
Space complexity: O(n)*/
public class Solution {
    public bool GcdSort(int[] nums) {
        int m = nums.Max();
        int n = nums.Length;
    
    int[] p = Enumerable.Range(0, m + 1).ToArray();
   // iota(begin(p), end(p), 0);
  
    int find(int x) {
      return p[x] == x ? x : (p[x] = find(p[x]));
    };
  
    foreach (int x in nums)
      for (int d = 2; d <= Math.Sqrt(x); ++d)
        if (x % d == 0)
          p[find(x)] = p[find(x / d)] = find(d);
 
   int[] sorted = nums.OrderBy(x => x).ToArray();
    //Array.Sort(sorted);not working
    for (int i = 0; i < n; ++i)
      if (find(sorted[i]) != find(nums[i])) 
        return false;
 
    return true;
    }
}
// 1995. Count Special Quadruplets
/*Solution 1: Brute force (224ms)
Enumerate a, b, c, d.

Time complexity: O(C(n, 4)) = O(n4/24)
Space complexity: O(1)*/
public class Solution {
    public int CountQuadruplets(int[] nums) {
        int n = nums.Length;
    int ans = 0;
    for (int a = 0; a < n; ++a)
      for (int b = a + 1; b < n; ++b)
        for (int c = b + 1; c < n; ++c)
          for (int d = c + 1; d < n; ++d)
            ans += Convert.ToInt32(nums[a] + nums[b] + nums[c] == nums[d]);
    return ans;
    }
}
/*a + b + c = d = > a + b = d - c
Break array into two parts[0, i - 1] and [i, n -1]
for each i,
step 1: calculate all possible d - c and put them in a hashMap called diffCount .
d - c = nums[j] - nums[i]. for all j [i + 1, n - 1]
step 2: calculate all possible a + b in the 1st part. Then check if any a + b in the hashMap diffCount
a + b = nums[j] + nums[i - 1], for all j [0, i - 2]

Time complexity: O(n^2)
Space complexity: O(n^2)*/
public class Solution {
    public int CountQuadruplets(int[] nums) {
       int result = 0;
        int n = nums.Length;
        Dictionary<int, int> diffCount = new Dictionary<int, int>();
        for (int i = n - 2; i >= 1; i--) {
            for (int j = i + 1; j < n; j++) {
                int num = nums[j] - nums[i];
                diffCount[num] = diffCount.GetValueOrDefault(num, 0) + 1;
            }
            
            for (int j = i - 2; j >= 0; j--) {
                int num = nums[j] + nums[i - 1];
                result += diffCount.GetValueOrDefault(num, 0);
            }
        }
        return result;  
  }
}
/*Solution 3: Dynamic frequency table (29ms)
Similar to 花花酱 LeetCode 1. 
Two Sum, we dynamically add elements (from right to left) into the hashtable.

Time complexity: O(n3/6)
Space complexity: O(n)

*/
public class Solution {
    public int CountQuadruplets(int[] nums) {
         int kMax = 100;
    int n = nums.Length;
    int ans = 0;
    int[] m = new int[kMax + 1];    
    // for every c we had seen, we can use it as target (nums[d]).
    for (int c = n - 1; c >= 0; ++m[nums[c--]])
      for (int a = 0; a < c; ++a)
        for (int b = a + 1; b < c; ++b) {
          int t = nums[a] + nums[b] + nums[c];
          if (t > kMax) continue;
          ans += m[t];
        }
    return ans;
  }
}

// 1991. Find the Middle Index in Array
/*Solution: Pre-compute + prefix sum
Pre-compute the sum of entire array. 
We scan the array and accumulate prefix sum and 
we can compute the sum of the rest of array.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int FindMiddleIndex(int[] nums) {
        int sum = nums.Sum();
    for (int i = 0, cur = 0; i < nums.Length; ++i) {
      if (cur == sum - cur - nums[i]) return i;
      cur += nums[i];
    }
    return -1;
    }
}

// 1985. Find the Kth Largest Integer in the Array
/*Solution: nth_element / quick selection
Use std::nth_element to find the k-th largest element. 
When comparing two strings, compare their lengths first and 
compare their content if they have the same length.

Time complexity: O(n) on average
Space complexity: O(1)

*/
public class Solution {
    public string KthLargestNumber(string[] nums, int k) {
        /*nth_element(begin(nums), begin(nums) + k - 1, end(nums), [](const auto& a, const auto& b) {
      return a.length() == b.length() ? a > b : a.length() > b.length();      
    });*/
        Array.Sort(nums,(a,b) => a.Length == b.Length ? b.CompareTo(a) : b.Length - a.Length);
    return nums[k - 1];
    }
}
public class Solution {
    public string KthLargestNumber(string[] nums, int k) {
        /*nth_element(begin(nums), begin(nums) + k - 1, end(nums), [](const auto& a, const auto& b) {
      return a.length() == b.length() ? a > b : a.length() > b.length();      
    });*/
       return nums.OrderBy(x => x.Length).ThenBy(x => x).ElementAt(nums.Length - k);
    }
}
// Linq
public class Solution {
    public string KthLargestNumber(string[] nums, int k) {
        return nums.OrderByDescending(o => o.Length)
                    .ThenByDescending(o => o)
                    .Skip(k-1)
                    .FirstOrDefault();
    }
}

public class Solution {
    public string KthLargestNumber(string[] nums, int k) => nums
      .OrderByDescending(item => item.Length)
      .ThenByDescending(item => item)
      .Skip(k - 1)
      .First();
}

// 1984. Minimum Difference Between Highest and Lowest of K Scores
/*Solution: Sliding Window
Sort the array, to minimize the difference, 
k numbers must be consecutive (i.e, from a subarray). 
We use a sliding window size of k and try all possible subarrays.
Ans = min{(nums[k – 1] – nums[0]), (nums[k] – nums[1]), … (nums[n – 1] – nums[n – k])}

Time complexity: O(nlogn)
Space complexity: O(1)*/
public class Solution {
    public int MinimumDifference(int[] nums, int k) {
        int ans = int.MaxValue;
    Array.Sort(nums);
    for (int i = k - 1; i < nums.Length; ++i)
      ans = Math.Min(ans, nums[i] - nums[i - k + 1]);
    return ans;
    }
}

// 1980. Find Unique Binary String
/*Solution 1: Hashtable
We can use bitset to convert between integer and binary string.

Time complexity: O(n2)
Space complexity: O(n2)*/
public class Solution {
    public string FindDifferentBinaryString(string[] nums) {
        int n = nums.Length;
    List<int> seen = nums.Select(o => Convert.ToInt32(o, 2)).ToList();
    for (int i = 0; i < (1 << n); ++i)
      if (!seen.Contains(i)) return Convert.ToString(i, 2).PadLeft(nums.Length, '0');
    return "";
       
    }
}
/*Solution 2: One bit a time
Let ans[i] = ‘1’ – nums[i][i], s.t. 
ans is at least one bit different from any strings.

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public string FindDifferentBinaryString(string[] nums) {
        int n = nums.Length;
    String ans = new String('0', n);

    for (int i = 0; i < n; i++){

      ans = ans.Remove(i, 1).Insert(i,Convert.ToChar('1' - nums[i][i] + '0').ToString());

    }
        
    return ans;
    }
}

public class Solution {
    public string FindDifferentBinaryString(string[] nums) {
        int n = nums.Length;
    String ans = new String('0', n);

    for (int i = 0; i < n; i++){
        char c = Convert.ToChar('1' - nums[i][i] + '0');
       ans = ans.Remove(i, 1);
        ans = ans.Insert(i,c.ToString());
    }
        
    return ans.ToString();
    }
}

public class Solution {
    public string FindDifferentBinaryString(string[] nums) {
        int n = nums.Length;
    StringBuilder ans = new StringBuilder();
    for (int i = 0; i < n; ++i)

      ans.Append(nums[i][i] == '0' ? "1" : "0");
    return ans.ToString();
    }
}
// 1979. Find Greatest Common Divisor of Array
/*Solution:
Use std::minmax_element and std::gcd

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int FindGCD(int[] nums) {
       /* auto p = minmax_element(begin(nums), end(nums));     
        return gcd(*p.first, *p.second);*/
        int max = nums.Max();
        int min = nums.Min();
    
        // Find the min and max from array
        return gcd(min, max);
    }
	
	private int gcd(int a, int b) {
		  return b > 0 ? gcd(b, a % b) : a;
    }
}

public class Solution {
    public int FindGCD(int[] nums) {
       /* auto p = minmax_element(begin(nums), end(nums));     
        return gcd(*p.first, *p.second);*/
        int max = nums.Max();
        int min = nums.Min();
    
        // Find the min and max from array
        return gcd(min, max);
    }
	
	private int gcd(int a, int b) {
		  if (b == 0) return a;
		  return gcd(b, a % b);
    }
}

// 1912. Design Movie Rental System
/*Solution: Hashtable + TreeSet
We need three containers:
1. movies tracks the {movie -> price} of each shop. This is readonly to get the price of a movie for generating keys for treesets.
2. unrented tracks unrented movies keyed by movie id, value is a treeset ordered by {price, shop}.
3. rented tracks rented movies, a treeset ordered by {price, shop, movie}

Note: By using array<int, 3> we can unpack values like below:
array<int, 3> entries; // {price, shop, movie}
for (const auto [price, shop, moive] : entries)
…

Time complexity:
Init: O(nlogn)
rent / drop: O(logn)
search / report: O(1)

Space complexity: O(n)*/
public class MovieRentingSystem {
   private Dictionary<(int, int), int> movies= new Dictionary<(int, int), int>();// {shop -> movie} -> price
    private Dictionary<int, SortedSet<(int price, int shop)>> unrented= new Dictionary<int,SortedSet<(int, int)>>(); // movie -> {price, shop}
    private SortedSet<(int price, int shop, int movie)> rented = new SortedSet<(int price, int shop, int movie)>(); // {price, shop, movie}

    public MovieRentingSystem(int n, int[][] entries) {
         foreach (var e in entries)
        {
            (int shop, int movie, int price) = (e[0], e[1], e[2]);
             movies.Add((shop, movie),  (price) );
            if (!unrented.ContainsKey(movie))
            {
                unrented.Add(movie, new SortedSet<(int price, int shop)>() { (price, shop) });  
            }
            else
            {
                unrented[movie].Add((price, shop));
            }
         }
    }
    
    public IList<int> Search(int movie) {
         IList<int> shops = new List<int>();
        if (!unrented.ContainsKey(movie))
        {
            return Array.Empty<int>();
        }
        
        foreach (var (price, shop) in unrented[movie]) {
          shops.Add(shop);
          if (shops.Count == 5) break;
        }
        return shops;
    }
    
    public void Rent(int shop, int movie) {
        
        
        int price = movies[(shop,movie)];
        unrented[movie].Remove((price, shop));
        rented.Add((price, shop, movie));
    }
    
    public void Drop(int shop, int movie) {
      
        int price = movies[(shop,movie)];
        rented.Remove((price, shop, movie));
        unrented[movie].Add((price, shop));
    }
    
    public IList<IList<int>> Report() {
        IList<IList<int>> ans = new List<IList<int>>();
        foreach (var (price, shop, movie) in rented) {
      ans.Add(new List<int>(){shop, movie});
      if (ans.Count == 5) break;
    }
    return ans;
        
    }
}

/**
 * Your MovieRentingSystem object will be instantiated and called as such:
 * MovieRentingSystem obj = new MovieRentingSystem(n, entries);
 * IList<int> param_1 = obj.Search(movie);
 * obj.Rent(shop,movie);
 * obj.Drop(shop,movie);
 * IList<IList<int>> param_4 = obj.Report();
 */

public class MovieRentingSystem {
    Dictionary<int, SortedSet<(int price, int shop)>> movies = new Dictionary<int, SortedSet<(int, int)>>();
    Dictionary<(int, int), int> movieShopMapToPrice= new Dictionary<(int, int), int>();
    Dictionary<int, HashSet<int>> rented= new Dictionary<int, HashSet<int>>(); 
    SortedSet<(int price, int shop, int movie)> cheapestRentedMovies = new SortedSet<(int price, int shop, int movie)>();

    public MovieRentingSystem(int n, int[][] entries) {
        foreach (var entry in entries)
        {
            (int shop, int movie, int price) = (entry[0], entry[1], entry[2]);
            if (!movies.ContainsKey(movie))
            {
                movies.Add(movie, new SortedSet<(int price, int shop)>() { (price, shop) });                        
            }

            else
            {
                movies[movie].Add((price, shop));
            }

            movieShopMapToPrice.Add((movie, shop), price);
        }
    }
    
    public IList<int> Search(int movie) {
         IList<int> result = new List<int>();
        if (!movies.ContainsKey(movie))
        {
            return Array.Empty<int>();
        }

        var current = movies[movie];
        return current.Take(5).Select(x => x.shop).ToList();
    }
    
    public void Rent(int shop, int movie) {
        if (!movies.ContainsKey(movie))
        {
            return;
        }

        if (rented.ContainsKey(movie))
        {
            rented[movie].Add(shop);
        }

        else
        {
            rented[movie] = new HashSet<int>() { shop };
        }

        int price = movieShopMapToPrice[(movie, shop)];
        cheapestRentedMovies.Add((price, shop, movie));
        movies[movie].Remove((price, shop));
    }
    
    public void Drop(int shop, int movie) {
        rented[movie].Remove(shop);
        int price = movieShopMapToPrice[(movie, shop)];
        cheapestRentedMovies.Remove((price, shop, movie));
        movies[movie].Add((price, shop));
    }
    
    public IList<IList<int>> Report() {
        IList<IList<int>> result = new List<IList<int>>();

        result = cheapestRentedMovies.Take(5)
            .Select(r => new int[] { r.shop, r.movie } as IList<int>).ToList();

        return result;
    }
}

/**
 * Your MovieRentingSystem object will be instantiated and called as such:
 * MovieRentingSystem obj = new MovieRentingSystem(n, entries);
 * IList<int> param_1 = obj.Search(movie);
 * obj.Rent(shop,movie);
 * obj.Drop(shop,movie);
 * IList<IList<int>> param_4 = obj.Report();
 */

 // 1975. Maximum Matrix Sum
 /*Solution: Math
Count the number of negative numbers.
1. Even negatives, we can always flip all the negatives to positives. 
ans = sum(abs(matrix)).
2. Odd negatives, there will be one negative left, 
we found the smallest abs(element) and let it become negative. 
ans = sum(abs(matrix))) – 2 * min(abs(matrix))

Time complexity: O(n2)
Space complexity: O(1)

*/
 public class Solution {
    public long MaxMatrixSum(int[][] matrix) {
        int n = matrix.Length;
   long ans = 0;
    int count = 0;
    int lo = Int32.MaxValue;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        ans += Math.Abs(matrix[i][j]);
        lo = Math.Min(lo, Math.Abs(matrix[i][j]));
        count += Convert.ToInt32(matrix[i][j] < 0);          
      }    
    return ans - (count & 1) * 2 * lo;
    }
}

// 1974. Minimum Time to Type Word Using Special Typewriter
/*Solution: Clockwise or Counter-clockwise?
For each pair of (prev, curr), choose the shortest distance.
One is abs(p – c), another is 26 – abs(p – c).
Don’t forget to add 1 for typing itself.

Time complexity: O(n)
Space complexity: O(1)
*/
public class Solution {
    public int MinTimeToType(string word) {
        int ans = 0;
    char p = 'a';
    foreach (char c in word) {
      ans += 1 + Math.Min(Math.Abs(c - p), 26 - Math.Abs(c - p));
      p = c;
    }
    return ans;
    }
}
// 1967. Number of Strings That Appear as Substrings in Word
public class Solution {
    public int NumOfStrings(string[] patterns, string word)  {
        var query = from s in patterns
                where word.Contains(s)
                select s;
        return query.Count();
    }
}

public class Solution {
    public int NumOfStrings(string[] patterns, string word) => patterns.Count(p => word.Contains(p));
}

/*Solution: Brute Force
We can use count_if for 1-liner.

Time complexity: O(m*n)
Space complexity: O(1)
*/
public class Solution {
    public int NumOfStrings(string[] patterns, string word) {
         /*return count_if(begin(patterns), end(patterns), [&](string& p) {
      return word.find(p) != string::npos;
    });*/
        int res = 0;
        foreach(String p in patterns){
           if (word.IndexOf(p) >= 0) { res++; } // C# IndexOf
        }
        return res; 
        
    }
}

public class Solution {
    public int NumOfStrings(string[] patterns, string word) {
         /*return count_if(begin(patterns), end(patterns), [&](string& p) {
      return word.find(p) != string::npos;
    });*/
        int res = 0;
        foreach(String p in patterns){
            if(word.Contains(p)){
                res ++;
            }
        }
        return res; 
        
    }
}

// 1962. Remove Stones to Minimize the Total
/* Heap Solution, O(klogn)
Explanation
Use a max heap.
Each time pop the max value a,
remove a / 2 from the number of stones res
and push back the ceil half a - a / 2 to the heap.
Repeat this operation k times.


Complexity
Time O(n + klogn)
Space O(n)*/
public class Solution {
    public int MinStoneSum(int[] piles, int k) {
        int total = piles.Sum();
    PriorityQueue<int, int> q = new PriorityQueue<int,int>(Comparer<int>.Create((a, b)=>b - a));//(begin(piles), end(piles));
        foreach (int a in piles) {
            q.Enqueue(a,a);
            //res += a;
        }
    while (k-- > 0) {
      int curr = q.Peek(); q.Dequeue();
      int remove = curr / 2;
      total -= remove;
      q.Enqueue(curr - remove, curr - remove);
    }
    return total;
        /*PriorityQueue<Integer> pq = new PriorityQueue<>((a, b)->b - a);
        int res = 0;
        for (int a : A) {
            pq.add(a);
            res += a;
        }
        while (k-- > 0) {
            int a = pq.poll();
            pq.add(a - a / 2);
            res -= a / 2;
        }
        return res;*/
    }
}

// 1961. Check If String Is a Prefix of Array
/*Solution: Concat and test
Time complexity: O(n)
Space complexity: O(n)
*/
public class Solution {
    public bool IsPrefixString(string s, string[] words) {
        string t = String.Empty;
    foreach (string w in words) {
      t += w;
      if (t.Length >= s.Length) break;
    }
    return t == s;
    }
}

// 1957. Delete Characters to Make Fancy String
public class Solution {
    public string MakeFancyString(string s) {
       if(s.Length <= 2) return s;
        
        StringBuilder sb = new StringBuilder();
        sb.Append(s.Substring(0, 2));
        
        for(int i = 2; i < s.Length; i++)
        {
            if(s[i - 2] == s[i - 1] && s[i - 1] == s[i]) continue;
            else sb.Append(s[i]);
        }
        
        return sb.ToString();
    }
}
/*time: O(n) iterate each character once
space: O(n) for the result*/
public class Solution {
    public string MakeFancyString(string s) {
       int idx = 0, freq = 0;
        char currChar = ' ';
        StringBuilder sb = new StringBuilder();
        foreach(var c in s)
        {
            if(c != currChar)
            {
                sb.Append(c);
                currChar = c;
                freq = 1;
            }
            else if(c == currChar)
            {
                if(freq < 2)
                {
                    sb.Append(c);
                    freq++;
                }           
            }
        }
        
        return sb.ToString();
    }
}
// 1953. Maximum Number of Weeks for Which You Can Work
public class Solution {
    public long NumberOfWeeks(int[] milestones) {
       long sum =  0; Array.ForEach(milestones, i => sum += i);
        //long sum = milestones.Sum();// not working
    long longest = milestones.Max();
    return Math.Min(sum, 2 * (sum - longest) + 1);
    }
}

// 1952. Three Divisors
/*Optimization
Only need to enumerate divisors up to sqrt(n). Special handle for the d * d == n case.

Time complexity: O(sqrt(n))
Space complexity: O(1)*/
public class Solution {
    public bool IsThree(int n) {
         int c = 0;
    for (int i = 1; i <= Math.Sqrt(n); ++i)
      if (n % i == 0)
        c += 1 + Convert.ToInt32(i * i != n);
    return c == 3;

    }
}
/*Solution: Enumerate divisors.
Time complexity: O(n)
Space complexity: O(1)
*/
public class Solution {
    public bool IsThree(int n) {
        int c = 0;
    for (int i = 1; i <= n; ++i)
      if (n % i == 0) ++c;        
    return c == 3;
    }
}

// 1946. Largest Number After Mutating Substring
/*As the mutation needs to be continuous, 
you need to search from the most significant digit towards 
the least significant digit and greedily try to exchange with a higher value 
if possible. 
If possible, then this would be the largest number as you were already searching
 from the most significant digit, so just return the new number.*/
public class Solution {
    public string MaximumNumber(string num, int[] change) {
       // var sb = new StringBuilder(num);
        /* int n = num.Length;
    for (int i = 0; i < n; ++i)
      if (num[i] - '0' < change[num[i] - '0']) {
        for (int j = i; j < n && num[j] - '0' <= change[num[j] - '0']; ++j){
             //string changeDigit = Convert.ToChar(change[num[i]-'0'] + '0').ToString(); 
           num = num.Remove(j, 1).Insert(j,Convert.ToChar(change[num[i]-'0'] + '0').ToString()); 
            //num[j] = change[num[j] - '0'] + '0';
        }
        break;
      }
    return num;*/
        
        var arr = num.ToCharArray().Select(x => x - '0').ToArray();
        for (var i = 0; i < arr.Length; i++) {
            if (change[arr[i]] <= arr[i]) {
                continue;
            }
            
            var j = i;
            while (j < num.Length && change[arr[j]] >= arr[j]) {
                arr[j] = change[arr[j]];
                j++;
            }
            
            return string.Join("", arr);
        }
        
        return num;
    }
}
/*Look for a digit that is greater when changed, 
when you find that digit change until a digit is not greater than or equal to. 
If no changes, return original string.*/
public class Solution {
    public string MaximumNumber(string num, int[] change) {
        
        var digits = num.ToCharArray();
        
        for(int i = 0; i < digits.Length; i++){
            int test = digits[i]-'0';
            
			// find the first digit greater than and change it
            if( test < change[test]){
                digits[i] = (char)(change[test] + '0');
				
				// change subsequent digits greater than or equal to
                for(int j = i + 1; j < digits.Length; j++){
                    int test2 = digits[j] - '0';
                    if(test2 <= change[test2]){
                        digits[j] = (char)(change[test2] + '0');
                    }else{
                        break;
                    }
                }
                return new string(digits);
            }
        }
        return num;
    }
}

// 1945. Sum of Digits of String After Convert
/*Solution: Simulation
Time complexity: O(klogn)
Space complexity: O(1)*/
public class Solution {
    public int GetLucky(string s, int k) {
        int ans = 0;
    foreach (char c in s) {
      int n = c - 'a' + 1;
      ans += n % 10;
      ans += n / 10;
    }
    while (--k > 0) {
      int n = ans;
      ans = 0;
      while (n > 0) {
        ans += n % 10;
        n /= 10;
      }
    }
    return ans;
    }
}

// 1942. The Number of the Smallest Unoccupied Chair
public class Solution {
    public int SmallestChair(int[][] times, int targetFriend) {

       var chairs = new int[times.Length];
        
        int friendStart = times[targetFriend][0];
        
        var sortedTimes = times.Where(x => x[0] <= friendStart).OrderBy(x => x[0]);
        
        foreach(var current in sortedTimes){
            for(int j = 0; j < chairs.Length; j++){
                if(chairs[j] <= current[0]){
                    if(friendStart == current[0]) return j;
                    chairs[j] = current[1];
                    break;
                }
            }
        }
        
        return -1;

    }
}
/*Solution: Treeset + Simulation
Use a treeset to track available chairs, sort events by time.
note: process leaving events first.

Time complexity: O(nlogn)
Space complexity: O(n)
*/
public class Solution {
    public int SmallestChair(int[][] times, int targetFriend) {

        
        int targetStart = times[targetFriend][0];
        Array.Sort(times, (x,y) => x[0] - y[0]);
        SortedSet<(int time, int chair)> chairs = new SortedSet<(int time, int chair)>();
        SortedSet<int> avail = new SortedSet<int>();
        
        for(int i = 0; i < times.Length; i++) {
            avail.Add(i);
        }
        
        for(int i = 0; i < times.Count(); i++) {
            int[] friend = times[i];
            
            if(chairs.Count() > 0) {
                var fc = chairs.First();
                while(fc.time <= friend[0]) {
                    avail.Add(fc.chair);
                    chairs.Remove((fc.time, fc.chair));
                    if(chairs.Count() <= 0) break;
                    fc = chairs.First();
                }
            }

            if(friend[0] == targetStart) break;

            chairs.Add((friend[1], avail.First()));
            avail.Remove(avail.First());                    
        }
        
        return avail.First();
    }
}

// 1941. Check if All Characters Have Equal Number of Occurrences
/*Solution: Hashtable
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public bool AreOccurrencesEqual(string s) {
         int[] m = new int[26];
    int maxCount = 0;
    foreach (char c in s)
      maxCount = Math.Max(maxCount, ++m[c - 'a']);
         foreach (int c in m)
            if (c == 0 || c == maxCount) return true; 

        return false;
        /*
    return all_of(begin(m), end(m), [maxCount](int c) -> bool {
      return c == 0 || c == maxCount;
    });
        
        int[] freq = new int[26];
        
        for (int i = 0; i < s.length(); i++) freq[s.charAt(i)-'a']++;

        int val = freq[s.charAt(0) - 'a'];
        foreach (int c in m)
            if (c == 0 || c == maxCount) return true; 

        return false;*/
    }
}

// 1936. Add Minimum Number of Rungs
/*Solution: Math
Check two consecutive rungs, if their diff is > dist, 
we need insert (diff – 1) / dist rungs in between.
ex1 5 -> 11, diff = 6, dist = 2, (diff – 1) / dist = (6 – 1) / 2 = 2. => 5, 7, 9, 11.
ex2 0 -> 3, diff = 3, dist = 1, (diff – 1) / dist = (3 – 1) / 1 = 2 => 0, 1, 2, 3

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int AddRungs(int[] rungs, int dist) {
        int ans = 0;
    for (int i = 0; i < rungs.Length; ++i)
      ans += (rungs[i] - (i > 0 ? rungs[i - 1] : 0) - 1) / dist;    
    return ans;
    }
}
// 1935. Maximum Number of Words You Can Type
// C# One-line LINQ
public class Solution {
    public int CanBeTypedWords(string text, string brokenLetters) => text.Split(' ').Count(w => !w.Intersect(brokenLetters).Any());        
}

public class Solution {
    public int CanBeTypedWords(string text, string brokenLetters) {
      List<string>  s1 = text.Split(' ').ToList();
    for(int i = 0 ; i < brokenLetters.Length ; i++ ){
         s1.RemoveAll(x=> x.Contains(brokenLetters[i]));
        }
    return s1.Count();
    }
}

public class Solution {
    public int CanBeTypedWords(string text, string brokenLetters) {
         string[] words = text.Split(' ');
        char[] broken = new char[brokenLetters.Length];
        broken = brokenLetters.ToCharArray();
        int canType = 0;
        
        foreach(var word in words) {
            if(word.IndexOfAny(broken) == -1) canType++;
        }
        
        return canType;
    }
}
/*Solution: Hashset / bitset
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int CanBeTypedWords(string text, string brokenLetters) {
       int res = 0;

        String[] separado = text.Split(' ');

        for (int a = 0; a < separado.Length; a++)
        {
        if (!brokenLetters.Any(separado[a].Contains))
        {
            res++;   
        }   
        } 
        return res;   
    }
}

public class Solution {
    public int CanBeTypedWords(string text, string brokenLetters) {string[] textArr = text.Split(' ');
        int count = textArr.Length;
        for(int i = 0; i < textArr.Length; i++) {
            for(int j = 0; j < brokenLetters.Length; j++) {
                if(count == 0) {
                    return count;
                } else if(textArr[i].Contains(brokenLetters[j])) {
                    count--;
                    break;
                }
            }
        }
        return count;
    }
}
// 1930. Unique Length-3 Palindromic Subsequences
/*Solution: Enumerate first character of a palindrome
For a length 3 palindrome, we just need to enumerate the first character c.
We found the first and last occurrence of c in original string and scan the middle part to see how many unique characters there.

e.g. aabca
Enumerate from a to z, looking for a*a, b*b, …, z*z.
For a*a, aabca, we found first and last a, in between is abc, which has 3 unique letters.
We can use a hastable or a bitset to track unique letters.

Time complexity: O(26*n)
Space complexity: O(1)*/
public class Solution {
    public int CountPalindromicSubsequence(string s) {
       var arr = s.ToCharArray();
        int count (char c){
      int l = Array.IndexOf(arr, c);
      int r = Array.LastIndexOf(arr, c);
      if (l == -1 || r == -1) return 0;
      //bitset<26> m;
        var m = new HashSet<int>();
      for (int i = l + 1; i < r; ++i)
        m.Add(arr[i] - 'a');
      return m.Count;
    };
    int ans = 0;
    for (char c = 'a'; c <= 'z'; ++c)
      ans += count(c);
    return ans;
    
    }
}
/*O(N) complexity
Points to be noted:

Palindrome string is of length 3. So only first and third character have to be same.
Duplicate strings are not allowed. So we have to get unique characters as a second character.
Algorithm:

Loop through all 'a' to 'z' characters. We will count palindrome starting with each character from 'a' to 'z'.
Get the first and last occurance of a character.
Count unique characters in between these two indices. These will be candidates for second character in our palindrome.
Keep adding count in final answer.*/
public class Solution {
    public int CountPalindromicSubsequence(string s) {
       var cnt = 0;
        var arr = s.ToCharArray();
        for (char c = 'a'; c <= 'z'; c++)
        {
            var st = Array.IndexOf(arr, c);
            var end = Array.LastIndexOf(arr, c);
            var hash = new HashSet<char>();
            for (var j = st + 1; j < end; j++)
            {
                hash.Add(arr[j]);
            }
            
            cnt += hash.Count;
        }
        
        return cnt;
    }
}

// 1929. Concatenation of Array
/*Solution: Pre-allocation
Pre-allocate an array of length 2 * n.
ans[i] = nums[i % n]

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int[] GetConcatenation(int[] nums) {
        int n = nums.Length;
    int[] ans = new int[2 * n];
    for (int i = 0; i < 2 * n; ++i)
      ans[i] = nums[i % n];
    return ans;
    }
}
// 1925. Count Square Sum Triples
/*Solution: Enumerate a & b
Brute force enumerates a & b & c, which takes O(n3). Just need to enumerate a & b and validate c.

Time complexity: O(n2)
Space complexity: O(1)

*/
public class Solution {
    public int CountTriples(int n) {
        int ans = 0;
    for (int a = 1; a <= n; ++a)
      for (int b = 1; b <= n; ++b) {
        int c = (int)Math.Sqrt(a * a + b * b);
        if (c <= n && c * c == a * a + b * b) ++ans;        
      }
    return ans;
    }
}
// 1922. Count Good Numbers
/*Solution: Fast Power
Easy to see that f(n) = (4 + (n & 1)) * f(n – 1), f(1) = 5

However, since n is huge, we need to rewrite f(n) as 4n/2 * 5(n+1)/2 
and use fast power to compute it.

Time complexity: O(logn)
Space complexity: O(1)

*/
public class Solution {
    int kMod = (int)1e9 + 7;
    public long modPow(long b, long n) {
    long ans = 1;
    while (n > 0) {
        if ((n & 1) > 0) ans = (ans * b) % kMod;
        b = (b * b) % kMod;
        n >>= 1;
    }
    return ans;
    }
    public int CountGoodNumbers(long n) {
        return (int)((modPow(4, n / 2) * modPow(5, (n + 1) / 2)) % kMod);
    }
}
// 1921. Eliminate Maximum Number of Monsters
/*Solution: Greedy
Sort by arrival time, and see how many we can eliminate.

Time complexity: O(nlogn)
Space complexity: O(n)*/
public class Solution {
    public int EliminateMaximum(int[] dist, int[] speed) {
        int n = dist.Length;
    int[] t = new int[n];
    for (int i = 0; i < n; ++i)
      t[i] = (dist[i] + speed[i] - 1) / speed[i];
    Array.Sort(t);    
    for (int i = 0; i < n; ++i)
      if (t[i] <= i) return i;
    return n;
    }
}

// 1920. Build Array from Permutation
/*Solution 1: Straight forward
Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int[] BuildArray(int[] nums) {
        int[] ans = new int[nums.Length];
    for (int i = 0; i < nums.Length; ++i)
      ans[i] = nums[nums[i]];
    return ans;
    }
}
/*Solution 2: Follow up: Inplace Encoding
Since nums[i] <= 1000, 
we can use low 16 bit to store the original value and high 16 bit for new value.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int[] BuildArray(int[] nums) {
        int n = nums.Length;
    for (int i = 0; i < n; ++i)
      nums[i] |= (nums[nums[i]] & 0xffff) << 16;
    for (int i = 0; i < n; ++i)
      nums[i] >>= 16;
    return nums;
    }
}
// 2122. Recover the Original Array
/*Solution: Try all possible k
Sort the array, we know that the smallest number nums[0] is org[0] – k, 
org[0] + k (nums[0] + 2k) must exist in nums. 
We try all possible ks. k = (nums[i] – nums[0]) / 2.

Then we iterate the sorted nums array as low, 
and see whether we can find low + 2k as high using a dynamic hashtable.

Time complexity: O(n2)
Space complexity: O(n)*/
public class Solution {
    public int[] RecoverArray(int[] nums) {
       int n = nums.Length;
    Array.Sort(nums);
    Dictionary<int, int> m = new Dictionary<int, int>();
    foreach (int x in nums) {
        m[x] = m.GetValueOrDefault(x,0)+1;}
        
     int[] check(int k) {      
      List<int> ans = new List<int>();
      Dictionary<int, int> cur = new Dictionary<int, int>(m);
      foreach (int x in nums) {
        if ( cur[x] == 0) continue;
        --cur[x];
       cur[x+k*2] = cur.GetValueOrDefault(x+k*2,0) - 1;
        if (  cur[x+k*2] < 0) return new int[]{};
          ans.Add(x + k);
      }
      return ans.ToArray();
    };
    for (int i = 1; i < n; i++) {
      if (nums[i] == nums[0] || ((nums[i] - nums[0]) & 1) > 0) continue;
       int k = (nums[i] - nums[0]) / 2;
      int[] ans = check(k);
      if (ans.Length != 0) return ans;
    }
    return new int[]{};
    
    }
}
/*from N = 1000, we know we can try something similar to O(n2)
we find out K is actually a limited number, 
it would be the difference between first element with all the rest number, one by one, when we have this list of k, we can try them one by one.

when we have a possible k to guess, 
we will see if both low (nums[i]) and high (nums[i] + 2 * k) exist, 
and we increase counter by 1 (here in code has use tmp array), 
if counter is N / 2 in the end, 
we will conclue that we find one possible answer.*/
public class Solution {
    int[] ans;
    public int[] RecoverArray(int[] nums) {
        int n = nums.Length/2;
        ans = new int[n];
        Array.Sort(nums);
        Dictionary<int, int> hm = new Dictionary<int, int>();
    foreach (int x in nums) {
        hm[x] = hm.GetValueOrDefault(x,0)+1;}
        
        for(int i=1; i<=n; i++){
            int k = (nums[i] - nums[0]);
            if(k!=0 && k%2==0){
                if(check(nums, new Dictionary<int, int>(hm), k)) return ans;
            }
        }
        return ans;
    }
    bool check(int[] nums,Dictionary<int, int> map, int k){
        int idx = 0;
        foreach(int num in nums){
            if(map.GetValueOrDefault(num,0) == 0) continue;
            if(map.GetValueOrDefault(num+k,0) == 0) return false;
            map[num] =map[num]-1;
            map[num+k]= map[num+k]-1;
            ans[idx++] = num+k/2;
        }
        return true;
    
   }
}

public class Solution {
    int[] ans;
    public int[] RecoverArray(int[] nums) {
        int n = nums.Length/2;
        ans = new int[n];
        Array.Sort(nums);
        Dictionary<int, int> m = new Dictionary<int, int>();
    foreach (int x in nums) {
        m[x] = m.GetValueOrDefault(x,0)+1;}
        
        bool check( int k){
         Dictionary<int, int> map = new Dictionary<int, int>(m);
        int idx = 0;
        foreach(int num in nums){
            if(map.GetValueOrDefault(num,0) == 0) continue;
            if(map.GetValueOrDefault(num+k,0) == 0) return false;
            map[num] =map[num]-1;
            map[num+k]= map[num+k]-1;
            ans[idx++] = num+k/2;
        }
        return true;
    
   }
        
        for(int i=1; i<=n; i++){
            int k = (nums[i] - nums[0]);
            if(k!=0 && k%2==0){
                if(check( k)) return ans;
            }
        }
        return ans;
    }
    
}
// 2121. Intervals Between Identical Elements
/*Solution: Math / Hashtable + Prefix Sum
For each arr[i], suppose it occurs in the array of total c times, 
among which k of them are in front of it and c – k – 1 of them are after it. 
Then the total sum intervals:
(i – j1) + (i – j2) + … + (i – jk) + (jk+1-i) + (jk+2-i) + … + (jc-i)
<=> k * i – sum(j1~jk) + sum(jk+1~jc) – (c – k – 1) * i

Use a hashtable to store the indies of each unique number in the array 
and compute the prefix sum for fast range sum query.

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public long[] GetDistances(int[] arr) {int n = arr.Length;
    Dictionary<int, List<long>> m = new Dictionary<int, List<long>>();
    int[] pos = new int[n];
    for (int i = 0; i < n; ++i) {
       if (!m.ContainsKey(arr[i]))
            {
                m[arr[i]] = new List<long>();
            }

            m[arr[i]].Add(i);
      pos[i] = m[arr[i]].Count - 1;
    }
    foreach (var (k, idx) in m){
            for(int i = 1; i < idx.Count; i++)
            {
                idx[i] = idx[i-1] + idx[i];
            }
    }
     
    long[] ans = new long[n];
    for (int i = 0; i < n; ++i) {
      List<long> sums = m[arr[i]];
      long k = Convert.ToInt64(pos[i]);
      long c = Convert.ToInt64(sums.Count);     
      if (k > 0) ans[i] += k * i - sums[(int)k - 1];
      if (k + 1 < c) ans[i] += (sums[(int)c - 1] - sums[(int)k]) - (c - k - 1) * i;
    }
    return ans;
    }
}
/*if we have [2,1,2,1,2,1,2,1,2], if we are looking for ans[4],

ans[4] = (4-0)+(4-2) + (6-4) + (8-4)

there are two parts, the ones before and after the 4th element.

before = (4-0)+(4-2) ,
after = (6-4) + (8-4)

let's look at before first.
before = (4-0) + (4-2) = 4 * 2 - (0+2)
we can see that (0+2) is the accumulate from the first element to the second. or to say, the indices summation before the ith element (here i is 4).

after = (6-4) + (8-4) = (6+8) - 4 * 2.
here 6+8 is the sum from (i+1)th to the last.*/
public class Solution {
    public long[] GetDistances(int[] arr) {
         int n = arr.Length;
            var res = new long[n];
			//group same value's indexes
            var dict=new Dictionary<int,List<long>>();
            for(int i = 0; i < n; ++i)
            {
                if (!dict.ContainsKey(arr[i])) dict.Add(arr[i], new List<long>());
                dict[arr[i]].Add(i);
            }
            foreach(var k in dict.Keys)
            {
                //init sum from index-0 to others
                var sum = dict[k].Sum(x => x - dict[k][0]);
                int count = dict[k].Count;
                for (int i = 0; i < count; ++i)
                {
                    res[dict[k][i]] = sum;
                    //when move from i to i+1, the gap between i to i+1 is (dict[k][i + 1] - dict[k][i])
                    //count-1-i nodes on right range [i+1,count-1], need subtract (count - 1 - i) * gap
                    //and i+1 nodes on left range [0,i], need add (i+1)*gap
                    if (i< dict[k].Count - 1)
                    {
                        sum -= (count - 1 - i) * (dict[k][i + 1] - dict[k][i]);
                        sum += (i + 1) * (dict[k][i + 1] - dict[k][i]);
                    }
                }
            }
            return res;
    }
}

public class Solution {
    public long[] GetDistances(int[] arr) {
        int n = arr.Length;
        Dictionary<int, List<int>> positionMap = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            if (!positionMap.ContainsKey(arr[i]))
            {
                positionMap[arr[i]] = new List<int>();
            }

            positionMap[arr[i]].Add(i);
        }

        long[] ans = new long[n];
        foreach(List<int> list in positionMap.Values)
        {
            long[] preSums = new long[list.Count + 1];
            for(int i = 0; i < list.Count; i++)
            {
                preSums[i + 1] = preSums[i] + list[i];
            }

            for(int i = 0; i < list.Count; i++)
            {
                long v = list[i];
                // pre
                ans[v] = v * (i+1) - preSums[i+1];
                // post
                ans[v] += preSums[list.Count] - preSums[i] - v * (list.Count - i);
            }
        }

        return ans;
    }
}

// 2120. Execution of All Suffix Instructions Staying in a Grid
/*Solution: Simulation
Time complexity: O(m2)
Space complexity: O(1)*/
public class Solution {
    public int[] ExecuteInstructions(int n, int[] startPos, string s) {
        int m = s.Length;
    int moves(int k) {
      int x = startPos[1];
      int y = startPos[0];
      for (int i = k; i < m; ++i) {
        switch (s[i]) {
          case 'R': x += 1; break;
          case 'D': y += 1; break;
          case 'L': x -= 1; break;
          case 'U': y -= 1; break;
        }        
        if (x < 0 || x == n || y < 0 || y == n) 
          return i - k;
      }
      return m - k;
    };
    int[] ans = new int[m];
    for (int i = 0; i < m; ++i)
      ans[i] = moves(i);
    return ans;
    }
}
// 2119. A Number After a Double Reversal
/*Solution: Math
The number must not end with 0 expect 0 itself.

e.g. 1230 => 321 => 123
e.g. 0 => 0 => 0

Time complexity: O(1)
Space complexity: O(1)*/
public class Solution {
    public bool IsSameAfterReversals(int num) {
        return num == 0 || num % 10 > 0;
    }
}

// 2117. Abbreviating the Product of a Range

// 2115. Find All Possible Recipes from Given Supplies
// Solution: Brute Force
public class Solution {
    public IList<string> FindAllRecipes(string[] recipes, IList<IList<string>> ingredients, string[] supplies) {
         int n = recipes.Length;
    HashSet<string> s = new HashSet<string>(supplies);
    IList<string> ans = new List<string>();
    int[] seen = new int[n];
    while (true) {      
      bool newSupply = false;  
      for (int i = 0; i < n; ++i) {
        if (seen[i] != 0) continue;
        bool hasAll = true;
        foreach (string ingredient in ingredients[i])
          if (!s.Contains(ingredient)) {
            hasAll = false; 
            break;
          }
        if (!hasAll) continue;
        ans.Add(recipes[i]);
        seen[i] = 1;
        s.Add(recipes[i]);
        newSupply = true;              
      }
      if (!newSupply) break;
    }
    return ans;
    }
   
}
// 2114. Maximum Number of Words Found in Sentences
/*Solution: Count spaces
Time complexity: O(sum(len(sentences[i]))
Space complexity: O(1)*/
public class Solution {
    public int MostWordsFound(string[] sentences) {
        int ans = 0;
    foreach (string s in sentences)
      ans = Math.Max(ans, s.Split(" ").Length);
    return ans;
    }
}

// 2111. Minimum Operations to Make the Array K-Increasing
/*Solution: Longest increasing subsequence
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
Space complexity: O(n/k)*/
public class Solution {
    public int KIncreasing(int[] arr, int k) {
        int LIS (List<int> nums) {
      List<int> lis = new List<int>();
      foreach(int x in nums)
        if (lis.Count == 0 || lis[lis.Count-1] <= x)
          lis.Add(x);
        else{
            int temp = upperBound(lis, x); 
            lis[temp] = x;
        }
         
      return lis.Count;
    };
    int n = arr.Length;
    int ans = 0;
    for (int i = 0; i < k; ++i) {
      List<int> cur = new List<int>();
      for (int j = i; j < n; j += k)
        cur.Add(arr[j]);
      ans += cur.Count - LIS(cur);
    }
    return ans;
    }
    public int upperBound( List<int> nums, int target ) {
    int l = 0, r = nums.Count - 1;

    while( l <= r ){
          int m = (l + r) / 2;
      if (nums[m] > target) {
        r = m - 1;
      } else {
        l = m + 1;
      }
    }

    return l;
  }
}

// 2110. Number of Smooth Descent Periods of a Stock
/*Solution: DP
Same as longest decreasing subarray.

dp[i] := length of longest smoothing subarray ends with nums[i].

dp[i] = dp[i – 1] + 1 if nums[i] + 1 = nums[i – 1] else 1

Time complexity: O(n)
Space complexity: O(n) -> O(1)*/
public class Solution {
    public long GetDescentPeriods(int[] prices) {
         int n = prices.Length;
    long[] dp = new long[n]; Array.Fill(dp, 1);
    long ans = 1;
    for (int i = 1; i < n; ++i) {
      if (prices[i] - prices[i - 1] == -1)
        dp[i] = dp[i - 1] + 1;    
      ans += dp[i];
    }
    return ans;
    }
}

// 835. Image Overlap
/*Solution: Hashtable of offsets
Enumerate all pairs of 1 cells (x1, y1) (x2, y2), 
the key / offset will be ((x1-x2), (y1-y2)), i.e how should we shift the image to have those two cells overlapped. Use a counter to find the most common/best offset.

Time complexity: O(n4) Note: 
this is the same as brute force / simulation method if the matrix is dense.
Space complexity: O(n2)*/
public class Solution {
    public int LargestOverlap(int[][] img1, int[][] img2) {
        int n = img1.Length;
    Dictionary<int, int> m = new Dictionary<int, int>();
    for (int y1 = 0; y1 < n; ++y1)
      for (int x1 = 0; x1 < n; ++x1)
        if (img1[y1][x1] > 0)
          for (int y2 = 0; y2 < n; ++y2) 
            for (int x2 = 0; x2 < n; ++x2)
              if (img2[y2][x2] > 0)                 
                m[(x1 - x2) * 100 + (y1 - y2)] = m.GetValueOrDefault((x1 - x2) * 100 + (y1 - y2), 0) + 1;
    int ans = 0;
    foreach (var (key, count) in m)
      ans = Math.Max(ans, count);
    return ans;
    }
}

// 722. Remove Comments
/*Parse the source string by string. If there is /* detected then ignore everything until / is detected:
If // is detected and / has not been detected then ignore the current string.
Time complexity O(S) where S is the length of source code.
Spacce complexity O(1): only using list of strings that need to be returned.*/
public class Solution {
    public IList<string> RemoveComments(string[] source) {
       IList<string> res = new List<string>();
        if(source.Length == 0) return res;
        bool block = false;
        string r  = "";
        foreach( string s in source)
        {
            for(int i = 0;i<s.Length;i++)
            {
                if(!block)
                {
                    if(s[i] == '/' && i+1 < s.Length && s[i+1] == '*')
                    {
                        block = true;
                        i++;
                        //Console.WriteLine(i + " " + s.Length);
                    }
                    else if(s[i] == '/' && i+1 < s.Length && s[i+1] == '/')
                    {
                        i++;
                        //while(i < s.Length )
                        break;
                    }
                    else{
                        r += s[i]; 
                    }
                }
                else{
                    if(s[i] == '*' && i+1 < s.Length && s[i+1] == '/' )//&& i-1 >= 0 && s[i-1] != '/')
                    {
                        block = false;
                        i++;
                        //Console.WriteLine(i + " " + s.Length);
                    }
                }
            }
            if(!block && r != "")
            {
                res.Add(r);
                r = "";
            }
        }
        
        return res;
    }
}

public class Solution {
    public IList<string> RemoveComments(string[] source) {
      IList<string> output = new List<string>();

    bool isBlock =false;
    StringBuilder builder = new StringBuilder();
    
    for(int pos = 0; pos < source.Length; ++pos)
    {
        if(!isBlock)    
        builder  = new StringBuilder();
           
        for(int index = 0; index < source[pos].Length; ++index)
        {
            if(isBlock && index < source[pos].Length - 1 && source[pos][index] == '*' && source[pos][index + 1] == '/')
            {
                isBlock = false;
                index ++;
            }
            else if(!isBlock)
            {
                if(index < source[pos].Length - 1 && source[pos][index] == '/' && source[pos][index + 1] == '/')
                {
                    index++;
                    break;
                }
                else if(index < source[pos].Length - 1 && source[pos][index] == '/' && source[pos][index + 1] == '*')
                {
                    isBlock = true;
                    index++;
                }
                else
                {
                    builder.Append(source[pos][index]);
                }
                    
            }
        }
        
        if(builder.ToString() != string.Empty && !isBlock)            
        output.Add(builder.ToString());
    }
    
    return output;
    }
}

/*When parsing, regular expression is often not a good soultion; 
here we have an exception: just two simple patterns (comments) to remove. 
We can

Join lines into a single string (text)
Match and remove comments with a help of regular expression
Split text back into lines of code.*/
using System.Text.RegularExpressions;
public class Solution {
    public IList<string> RemoveComments(string[] source) => Regex
        .Replace(string.Join("\n", source), @"(/\*(.|\n)*?\*/)|(//.*?$)", "", RegexOptions.Multiline)
        .Split("\n", StringSplitOptions.RemoveEmptyEntries)
        .ToList();
    
}


// 1915. Number of Wonderful Substrings
/*Explanation
Use a mask to count the current prefix string.
mask & 1 means whether it has odd 'a'
mask & 2 means whether it has odd 'b'
mask & 4 means whether it has odd 'c'
...

We find the number of wonderful string with all even number of characters.
Then we flip each of bits, 10 at most, and doing this again.
This will help to find string with at most one odd number of characters.

Complexity
Time O(10n), Space O(1024)*/
public class Solution {
    public long WonderfulSubstrings(string word) {
        long res = 0; long[] count = new long[1024];
        int cur = 0;
        count[0] = 1L;
        for (int i = 0; i < word.Length; ++i) {
            cur ^= 1 << (word[i] - 'a');
            res += count[cur]++;
            for (int j = 0; j < 10; ++j)
                res += count[cur ^ (1 << j)];
        }
        return res;
    }
}
/*Solution: Prefix Bitmask + Hashtable
Similar to 花花酱 LeetCode 1371. Find the Longest Substring Containing Vowels in Even Counts, we use a bitmask to represent the occurrence (odd or even) of each letter and use a hashtable to store the frequency of each bitmask seen so far.

1. “0000000000” means all letters occur even times.
2. “0000000101” means all letters occur even times expect letter ‘a’ and ‘c’ that occur odd times.

We scan the word from left to right and update the bitmask: bitmask ^= (1 << (c-‘a’)).
However, the bitmask only represents the state of the prefix, i.e. word[0:i], then how can we count substrings? The answer is hashtable. If the same bitmask occurs c times before, which means there are c indices that word[0~j1], word[0~j2], …, word[0~jc] have the same state as word[0~i] that means for word[j1+1~i], word[j2+1~i], …, word[jc+1~i], all letters occurred even times.
For the “at most one odd” case, we toggle each bit of the bitmask and check how many times it occurred before.

ans += freq[mask] + sum(freq[mask ^ (1 << i)] for i in range(k))

Time complexity: O(n*k)
Space complexity: O(2k)
where k = j – a + 1 = 10*/
public class Solution {
    public long WonderfulSubstrings(string word) {
        int l = 'j' - 'a' + 1;
        int[] count = new int[1 << l];
        count[0] = 1;
        int mask = 0;
        long ans = 0;
        foreach (char c in word) {
        mask ^= 1 << (c - 'a'); 
        for (int i = 0; i < l; ++i)
            ans += count[mask ^ (1 << i)]; // one odd.
        ans += count[mask]++; // all even.
        }
        return ans;
    }
}

// 1913. Maximum Product Difference Between Two Pairs
/*Solution: Greedy
Since all the numbers are positive, 
we just need to find the largest two numbers as the first pair 
and smallest two numbers are the second pair.

Time complexity: O(nlogn) / sorting, O(n) / finding min/max elements.
Space complexity: O(1)*/
public class Solution {
    public int MaxProductDifference(int[] nums) {
         int n = nums.Length;
    Array.Sort(nums);
    return nums[n - 1] * nums[n - 2] - nums[0] * nums[1];
    }
}
// 1910. Remove All Occurrences of a Substring
/*Solution: Simulation
Time complexity: O(n2/m)
Space complexity: O(n)*/
public class Solution {
    public string RemoveOccurrences(string s, string part) {
        while (true) {
      int i = s.IndexOf(part);
      if (i == -1) break;
            s = s.Substring(0, i) + s.Substring(i + part.Length);     
    }
    return s;
        
    }
}

// 1909. Remove One Element to Make the Array Strictly Increasing
/*Solution 1: Brute Force
Enumerate the element to remove and check.

Time complexity: O(n2)
Space complexity: O(n)*/
public class Solution {
    public bool CanBeIncreasing(int[] nums) {
        int n = nums.Length;
    bool check(int k) {
      List<int> arr = new List<int>(nums);
      arr.RemoveAt(k);
      for (int i = 1; i < n - 1; ++i)
        if (arr[i] <= arr[i - 1]) return false;
      return true;
    };
    for (int i = 0; i < n; ++i)
      if (check(i)) return true;
    return false;
       
    
    }
}
/*Solution 2: Brute Force
Enumerate the element to remove and check.

Time complexity: O(n2)
Space complexity: O(1)*/
public class Solution {
    public bool CanBeIncreasing(int[] nums) {
        int n = nums.Length;
        bool check(int k) {      
        for (int i = 1; i < n; ++i) {        
            int j = (i - 1 == k) ? (i - 2) : (i - 1);
            if (i != k && j >= 0 && nums[i] <= nums[j])          
            return false;
        }
        return true;
        };
        for (int i = 0; i < n; ++i)
        if (check(i)) return true;
        return false;
       
    }
}

// 2109. Adding Spaces to a String
/*Solution: Scan orignal string / Two Pointers
Just scan the original string and insert a space 
if the current index matched the front space index.

Time complexity: O(n)
Space complexity: O(m+n) / O(1)*/
// Traverse input from left to right.
public class Solution {
    public string AddSpaces(string s, int[] spaces) {
       StringBuilder ans = new StringBuilder();
        for (int i = 0, j = 0; i < s.Length; ++i) {
            if (j < spaces.Length && spaces[j] == i) {
                ans.Append(' ');
                ++j;
            }
            ans.Append(s[i]);
        }
        return ans.ToString();
    }
}

public class Solution {
    public string AddSpaces(string s, int[] spaces) {
        StringBuilder sb = new StringBuilder(s);
        
        for(int i=0; i < spaces.Length; i++) {
            sb.Insert(spaces[i] + i, " ");
        }
        
        return sb.ToString();
    }
}

// 2108. Find First Palindromic String in the Array
/*Solution: Brute Force
Enumerate each word and check whether it’s a palindrome or not.

Time complexity: O(n * l)
Space complexity: O(1)*/
public class Solution {
    public string FirstPalindrome(string[] words) {
        bool isPalindrome(string s) {
      int l = s.Length;
      for (int i = 0; i < l / 2; ++i)
        if (s[i] != s[l - i - 1]) return false;
      return true;
    };
    foreach (string word in words)
      if (isPalindrome(word)) return word;
    return "";
    }
}

// 2106. Maximum Fruits Harvested After at Most K Steps
/*Solution 2: Sliding Window
Maintain a window [l, r] such that the steps to cover [l, r] 
from startPos is less or equal to k.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MaxTotalFruits(int[][] fruits, int startPos, int k) {
         int steps(int l, int r) {
      if (r <= startPos)
        return startPos - l;
      else if (l >= startPos)
        return r - startPos;
      else
        return Math.Min(startPos + r - 2 * l, 2 * r - startPos - l);
    };
    int ans = 0;
    for (int r = 0, l = 0, cur = 0; r < fruits.Length; ++r) {
      cur += fruits[r][1];
      while (l <= r && steps(fruits[l][0], fruits[r][0]) > k)
        cur -= fruits[l++][1];      
      ans = Math.Max(ans, cur);
    }
    return ans;
    }
}
/*Solution 1: Range sum query
Assuming we can collect fruits in range [l, r], we need a fast query to compute the sum of those fruits.

Given startPos and k, we have four options:
1. move i steps to the left
2. move i steps to the left and k – i steps to the right.
3. move i steps to the right
4. move i steps to the right and k – i steps to the left.

We enumerate i steps and calculate maximum range [l, r] covered by each option, and collect all the fruit in that range.

Time complexity: O(m + k)
Space complexity: O(m)
where m = max(max(pos), startPos)
*/
public class Solution {
    public int MaxTotalFruits(int[][] fruits, int startPos, int k) {
        
    int max = fruits[0].Max();
    for(int i=0;i<fruits.Length;i++)
    {
        if(max < fruits[i].Max())
        {
            max = fruits[i].Max();
        }
    }
     int m = Math.Max(startPos, max);
    int[] sums = new int[m + 2];   
    for (int i = 0, j = 0; i <= m; ++i) {
      sums[i + 1] += sums[i];
      while (j < fruits.Length && fruits[j][0] == i)        
        sums[i + 1] += fruits[j++][1];      
    }    
    int ans = 0;
    for (int s = 0; s <= k; ++s) {
      if (startPos - s >= 0) {
        int l = startPos - s;
        int r = Math.Min(Math.Max(startPos, l + (k - s)), m);        
        ans = Math.Max(ans, sums[r + 1] - sums[l]);
      }
      if (startPos + s <= m) {
        int r = startPos + s;
        int l = Math.Max(0, Math.Min(startPos, r - (k - s)));
        ans = Math.Max(ans, sums[r + 1] - sums[l]);
      }
    }             
    return ans;
    }
}
// 2105. Watering Plants II
/*Solution: Simulation w/ Two Pointers

Simulate the watering process.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MinimumRefill(int[] plants, int capacityA, int capacityB) {
         int n = plants.Length;    
    int ans = 0;
    for (int l = 0, r = n - 1, A = capacityA, B = capacityB; l <= r; ++l, --r) {      
      if (l == r)
        return ans += Convert.ToInt32(Math.Max(A, B) < plants[l]);
      if ((A -= plants[l]) < 0) {
          A = capacityA - plants[l]; ++ans;  
      }
              
      if ((B -= plants[r]) < 0) {
          B = capacityB - plants[r]; ++ans;    
      }
         
    }
    return ans;
    }
}
// 2104. Sum of Subarray Ranges
/*Solution 0: Brute force [TLE]
Enumerate all subarrays, for each one, find min and max.

Time complexity: O(n3)
Space complexity: O(1)

Solution 1: Prefix min/max
We can use prefix technique to extend the array 
while keep tracking the min/max of the subarray.

Time complexity: O(n2)
Space complexity: O(1)*/
public class Solution {
    public long SubArrayRanges(int[] nums) {
        int n = nums.Length;    
    long ans = 0;
    for (int i = 0; i < n; ++i) {
      int lo = nums[i];
      int hi = nums[i];
      for (int j = i + 1; j < n; ++j) {
        lo = Math.Min(lo, nums[j]);
        hi = Math.Max(hi, nums[j]);
        ans += (hi - lo);
      }
    }
    return ans;
    }
}
/*Solution 2, O(n) Stack Solution
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
Space O(n)*/
public class Solution {
    public long SubArrayRanges(int[] A) {
       int n = A.Length, j, k;
        long res = 0;
        
        Stack<int> s = new Stack<int>();
        for (int i = 0; i <= n; i++) {
            while (s.Count != 0 && A[s.Peek()] > (i == n ? int.MinValue : A[i])) {
                j = s.Pop();
                k = s.Count == 0 ? -1 : s.Peek();
                res -= (long)A[j] * (i - j) * (j - k);

            }
            s.Push(i);
        }
        
        s.Clear();
        for (int i = 0; i <= n; i++) {
            while (s.Count != 0 && A[s.Peek()] < (i == n ? int.MaxValue : A[i])) {
                j = s.Pop();
                k = s.Count == 0 ? -1 : s.Peek();
                res += (long)A[j] * (i - j) * (j - k);

            }
            s.Push(i);
        }
        return res;
    }
}
/*Solution 2: Monotonic stack
This problem can be reduced to 花花酱 LeetCode 907. Sum of Subarray Minimums

Just need to run twice one for sum of mins and another for sum of maxs.

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public long SubArrayRanges(int[] nums) {
        int n = nums.Length;
        long sumOf(int[] nums, Func<int,int,bool> op){
        long ans = 0;
        Stack<int> stack = new Stack<int>();
        for(int i = 0; i <= n; i++){
            while(stack.Count > 0 && (i == n || op(nums[stack.Peek()], nums[i]))){
                int m = stack.Pop(), l = stack.Count > 0 ? stack.Peek() : -1;
                ans += (long)nums[m]*(m - l)*(i - m);
            }
            stack.Push(i);
        }
        return ans;
        }
       return sumOf(nums,(a,b) => { return a < b; }) - sumOf(nums, (a,b) => { return a > b; });
    
    }
}
/*C# monostack solution

Intuition
This problem is based on - 907. Sum of Subarray Minimums.
But here we calculate Maximun and Minimum Sum of Subarray and return difference.

Approach
2 Monostacks

Complexity
Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public long SubArrayRanges(int[] nums) {
       return sum(nums,(a,b) => { return a < b; }) - sum(nums, (a,b) => { return a > b; });
    }
    private long sum(int[] nums, Func<int,int,bool> comp){
        long sum = 0;
        Stack<int> stack = new Stack<int>();
        for(int i = 0; i <= nums.Length; i++){
            while(stack.Count > 0 && (i == nums.Length || comp(nums[stack.Peek()], nums[i]))){
                int mid = stack.Pop(), left = stack.Count > 0 ? stack.Peek() : -1, right = i;
                sum += (long)nums[mid]*(mid-left)*(right-mid);
            }
            stack.Push(i);
        }
        return sum;
    }
}

// 2103. Rings and Rods
// Dictionary and HashSet
public class Solution {
    public int CountPoints(string rings) {
         var dictionary = new Dictionary<int, HashSet<char>>();
	for(var i = 0; i < rings.Length; i += 2)
	{
		if(dictionary.ContainsKey(rings[i + 1]))
		{
			dictionary[rings[i + 1]].Add(rings[i]);
		}
		else
		{
			dictionary.Add(rings[i + 1], new HashSet<char>() {rings[i]});
		}
	}

	return dictionary.Count(x => x.Value.Count == 3);
    }
}

public class Solution {
    public int CountPoints(string rings) {
         Dictionary<int,HashSet<char>> m = new Dictionary<int,HashSet<char>>();
        for(int i=0;i<rings.Length;i=i+2){
            char c=rings[i];
            int index=(int)rings[i+1];
            if(m.ContainsKey(index)){
                 HashSet<char> x = m[index];
                x.Add(c);
                m[index] = x;
            }else{
                HashSet<char> x = new HashSet<char>();
                x.Add(c);
                m[index] = x;
            }
        }
        int count=0;
        foreach(HashSet<char> k in m.Values){
            if(k.Count==3) count++;
        }
        return count;
    }
}
// LINQ
public class Solution {
    public int CountPoints(string rings) {
         return Enumerable
            .Range(0, rings.Length / 2)
            .Select(x => rings.Substring(2 * x, 2))
            .Distinct()
            .GroupBy(x => x[1])			
            .Count(g => g.Count() == 3);
    }
}

// 2102. Sequentially Ordinal Rank Tracker
public class SORTracker {

     public class Record {
    public string Name; public int Score;
}
    SortedList<Record, Record> list;
    private int index = 0;
    public SORTracker() {
        list = new SortedList<Record, Record>(Comparer<Record>.Create((a, b) => a.Score == b.Score ? a.Name.CompareTo(b.Name) : a.Score < b.Score ? 1 : -1));
    }
    
    public void Add(string name, int score) {
        var newRecord = new Record(){Name = name, Score=score};
    list.Add(newRecord, newRecord);
    }
    
    public string Get() {
         return list.Values[index++].Name;
    }
}

/**
 * Your SORTracker object will be instantiated and called as such:
 * SORTracker obj = new SORTracker();
 * obj.Add(name,score);
 * string param_2 = obj.Get();
 */


/*Keep 2 heaps, one max and the other min. 
The idea is to have the index point in the middle of the heaps. 
So for example, if at the 4th index out of 10 items, 
there will be 3 items in the min heap, 
the result will be the top of the max heap (after the index).

for add O(logN), we push to the min heap (before the index), 
hen we take the top of the min heap and push it to the max heap. 
This will balance the 2 heaps.
for get O(logN), the answer is the top of the max heap 
and we move that item to the min heap to keep the balance.*/
 public class SORTracker {

     SortedSet<Tuple<string, int>> before, after;
    public SORTracker() {
        before = new SortedSet<Tuple<string, int>>(Comparer<Tuple<string, int>>.Create((a, b) =>
      a.Item2 == b.Item2 ? b.Item1.CompareTo(a.Item1) : a.Item2 - b.Item2
    ));
    after = new SortedSet<Tuple<string, int>>(Comparer<Tuple<string, int>>.Create((a, b) =>
      a.Item2 == b.Item2 ? a.Item1.CompareTo(b.Item1) : b.Item2 - a.Item2
    ));
    }
    
    public void Add(string name, int score) {
        before.Add(Tuple.Create(name, score));
    var first = before.First();
    before.Remove(first);
    after.Add(first);
    }
    
    public string Get() {
         var res = after.First();
    after.Remove(res);
    before.Add(res);
    return res.Item1;
    }
}

/**
 * Your SORTracker object will be instantiated and called as such:
 * SORTracker obj = new SORTracker();
 * obj.Add(name,score);
 * string param_2 = obj.Get();
 */
// 2101. Detonate the Maximum Bombs
/*Solution: Simulation w/ BFS
Enumerate the bomb to detonate, and simulate the process using BFS.

Time complexity: O(n3)
Space complexity: O(n)*/
public class Solution {
    public int MaximumDetonation(int[][] bombs) {int n = bombs.Length;                                                  
    bool check (int i, int j) {
      long x1 = bombs[i][0], y1 = bombs[i][1], r1 = bombs[i][2];
      long x2 = bombs[j][0], y2 = bombs[j][1], r2 = bombs[j][2];
      return ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) <= r1 * r1);          
    };
    int ans = 0;
    for (int s = 0; s < n; ++s) {
      int count = 0;
      Queue<int> q = new Queue<int>(); q.Enqueue(s);
      int[] seen = new int[n]; Array.Fill(seen, 0);
      seen[s] = 1;
      while (q.Count != 0) {
          ++count;
        int i = q.Peek(); q.Dequeue();
        for (int j = 0; j < n; ++j)
          if (check(i, j) && seen[j]++ == 0 ){
              q.Enqueue(j); 
          }
      }
      ans = Math.Max(ans, count);
    }
    return ans;
    }
}
// The main idea here is to take each bomb and check the number of bombs in its range.
public class Solution {
    public int MaximumDetonation(int[][] bombs) {int n = bombs.Length;    
        int ans = 0;
        //iterate through each bomb and keep track of max
        for(int i = 0; i<bombs.Length; i++){
            ans = Math.Max(ans, getMaxBFS(bombs, i));    
        }
        return ans;
    }
    
   private int getMaxBFS(int[][] bombs, int index){
        Queue<int> queue = new Queue<int>();
        bool[] seen = new bool[bombs.Length];
        
        seen[index] = true;
        queue.Enqueue(index);
        
        int count = 1; // start from 1 since the first added bomb can detonate itself
        
        while(queue.Count != 0){
            int currBomb = queue.Dequeue();
            for(int j = 0; j<bombs.Length; j++){ //search for bombs to detonate
                if(!seen[j] && isInRange(bombs[currBomb], bombs[j])){
                    seen[j] = true;
                    count++;
                    queue.Enqueue(j);
                }
            }
        }       
        return count;
    }
    
    //use the distance between two points formula
    //then check if curr bomb radius is greater than the distance; meaning we can detonate the second bombs
    private bool isInRange(int[] point1, int[] point2) {
        long dx = point1[0] - point2[0], dy = point1[1] - point2[1], radius = point1[2];
        long distance =  dx * dx + dy * dy;
        return distance <= radius * radius;  
	}
}

public class Solution {
    public int MaximumDetonation(int[][] bombs) {
       int max = 0;
		Dictionary<int, List<int>> graph = new Dictionary<int, List<int>>();

		for (int i = 0; i < bombs.GetLength(0); i++)
		{
			if (!graph.ContainsKey(i))
				graph[i] = new List<int>();

			for (int j = i + 1; j < bombs.GetLength(0); j++)
			{
				if (!graph.ContainsKey(j))
					graph[j] = new List<int>();

				if (CheckIfCirclesIntersact(bombs[i][0], bombs[i][1], bombs[j][0], bombs[j][1], bombs[i][2]))
					graph[i].Add(j);

				if (CheckIfCirclesIntersact(bombs[j][0], bombs[j][1], bombs[i][0], bombs[i][1], bombs[j][2]))
					graph[j].Add(i);
			}
		}

		for (int i = 0; i < bombs.GetLength(0); i++)
		{
			HashSet<int> set = new HashSet<int>() { i };
			BFS(graph, i, set);
			max = Math.Max(max, set.Count);
		}

		return max;
	}

	private void BFS(Dictionary<int, List<int>> graph, int idx, HashSet<int> set)
	{
		Queue<int> queue = new Queue<int>();

		queue.Enqueue(idx);

		while (queue.Count > 0)
		{
			int node = queue.Dequeue();

			foreach (var n in graph[node])
			{
				if (!set.Contains(n))
				{
					queue.Enqueue(n);
					set.Add(n);
				}
			}
		}
	}

	private bool CheckIfCirclesIntersact(int x1, int y1, int x2, int y2, int r1)
	{
		long dx = x1 - x2;
		long dy = y1 - y2;
		long radius = r1;

		long distSq = (dx * dx) + (dy * dy);
		long radSumSq = radius * radius;

		return distSq <= radSumSq;
	}
}

// 2100. Find Good Days to Rob the Bank
public class Solution {
    public IList<int> GoodDaysToRobBank(int[] security, int time) {
       int n = security.Length;
    int[] before = new int[n];
    int[] after = new int[n];
    for (int i = 1; i < n; ++i)
      before[i] = security[i - 1] >= security[i] ? before[i - 1] + 1 : 0;
    for (int i = n - 2; i >= 0; --i)
      after[i] = security[i + 1] >= security[i] ? after[i + 1] + 1 : 0;
    List<int> ans = new List<int>();
    for (int i = time; i + time < n; ++i)
      if (before[i] >= time && after[i] >= time)
        ans.Add(i);
    return ans.ToArray();       
        
    }
}
// 2099. Find Subsequence of Length K With the Largest Sum
public class Solution {
    public int[] MaxSubsequence(int[] nums, int k) {
         List<int> ans = new List<int>();
    List<int> s =  nums.OrderBy(x => -x).ToList();
   // Array.Sort(s,(a,b)=> b-a );
    Dictionary<int, int> m = new Dictionary<int, int>();
    for (int i = 0; i < k; ++i) m[s[i]] = m.GetValueOrDefault(s[i],0)+1;
    foreach (int x in nums) {
        m[x] = m.GetValueOrDefault(x,0) - 1;
         if (m[x] >= 0) 
        ans.Add(x);
    }
     
    return ans.ToArray();

    }
}

public class Solution {
    public int[] MaxSubsequence(int[] nums, int k) {
         List<int> n = nums.ToList();
        while (n.Count != k) {
            int mini = 0;
            for (int i = 0; i < n.Count; ++i)
                if (n[i] < n[mini])
                    mini = i;
            n.RemoveAt(mini);
        }
        
        return n.ToArray();;

    }
}
// linq solution
public class Solution {
    public int[] MaxSubsequence(int[] nums, int k) {
         return nums
        .Select((n, i) => new { Number = n, Index = i })
        .ToList()
        .OrderByDescending(n => n.Number)
        .Take(k)
        .OrderBy(n => n.Index)
        .Select(n => n.Number)
        .ToArray();

    }
}
// Linq
public class Solution {
    public int[] MaxSubsequence(int[] nums, int k) {
         return nums
             .Select((x, index) => new { x, index })
             .OrderBy(y => y.x)
             .Skip(nums.Length - k)
             .OrderBy(y=>y.index)
             .Select(y=>y.x)
             .ToArray();
    }
}
// Method 3: Quick Select
// Time: average O(n), worst O(n ^ 2), space: O(n).
public class Solution {
    public int[] MaxSubsequence(int[] nums, int k) {
         int n = nums.Length;
        int[] index = new int[n];
        for (int j = 0; j < n; ++j) {
            index[j] = j;
        }
        
        // Use Quick Select to put the indexes of the 
        // max k items to the left of index array. 
        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int idx = quickSelect(nums, index, lo, hi);
            if (idx < k) {
                lo = idx + 1;
            }else {
                hi = idx;
            }
        }
        
        // Count the occurrencs of the kth largest items
        // within the k largest ones.
        int kthVal = nums[index[k - 1]], freqOfkthVal = 0;
        foreach (int a in index.Take(k).ToArray()) {
            freqOfkthVal += nums[a] == kthVal ? 1 : 0;
        }
        
        // Greedily copy the subsequence into output array seq.
        int[] seq = new int[k];
        int i = 0;
        foreach (int num in nums) {
            if (num > kthVal || num == kthVal && freqOfkthVal-- > 0) {
                seq[i++] = num;
            }
        }
        return seq;

    }
    
    // Divide index[lo...hi] into two parts: larger and less than 
    // the pivot; Then return the position of the pivot;
    private int quickSelect(int[] nums, int[] index, int lo, int hi) {
        int pivot = index[lo];
        while (lo < hi) {
            while (lo < hi && nums[index[hi]] <= nums[pivot]) {
                --hi;
            }
            index[lo] = index[hi];
            while (lo < hi && nums[index[lo]] >= nums[pivot]) {
                ++lo;
            }
            index[hi] = index[lo];
        } 
        index[lo] = pivot;
        return lo;
    }
}

// 219. Contains Duplicate II
public class Solution {
    public bool ContainsNearbyDuplicate(int[] nums, int k) {
        Dictionary<int, int> m = new Dictionary<int, int>(); // num -> last index
    for (int i = 0; i < nums.Length; ++i) {
      if (i > k && m[nums[i - k - 1]] < i - k + 1)
        m.Remove(nums[i - k - 1]);
      if (m.ContainsKey(nums[i])) return true;
      m[nums[i]] = i;
    }
    return false;
    }
}

public class Solution {
    public bool ContainsNearbyDuplicate(int[] nums, int k) {
         HashSet<int> s = new HashSet<int>();
        for(int i = 0; i < nums.Length; i++){
            if(i > k) s.Remove(nums[i-k-1]);
            if(!s.Add(nums[i])) return true;
        }
        return false;
    }
}

// 2097. Valid Arrangement of Pairs
/*Solution: Eulerian trail
The goal of the problem is to find a Eulerian trail in the graph.

If there is a vertex whose out degree – in degree == 1 
which means it’s the starting vertex. 
Otherwise wise, the graph must have a Eulerian circuit thus 
we can start from any vertex.

We can use Hierholzer’s algorithm to find it.

Time complexity: O(|V| + |E|)
Space complexity: O(|V| + |E|)*/
public class Solution {
    public int[][] ValidArrangement(int[][] pairs) {
         Dictionary<int, Queue<int>> g = new Dictionary<int, Queue<int>> ();
    Dictionary<int, int> degree = new Dictionary<int, int>(); // out - in  
    foreach (int[] p in pairs) {
        g[p[0]] = g.GetValueOrDefault(p[0], new Queue<int>());
        degree[p[0]] = degree.GetValueOrDefault(p[0],0);
        degree[p[1]] = degree.GetValueOrDefault(p[1],0);
      g[p[0]].Enqueue(p[1]);
      ++degree[p[0]];
      --degree[p[1]];
    }
    
    int s = pairs[0][0];
    foreach (var (u, d) in degree)
      if (d == 1) s = u;
    
    int[][] ans = new int[pairs.Length][];
    int i = 0;
    void dfs(int u) {
      while (g.ContainsKey(u) && g[u].Count != 0) {
        int v = g[u].Peek(); g[u].Dequeue();
        dfs(v);
        ans[i] = new int[2]{u, v};i++;
      }
    };
    dfs(s);

    return ans.Reverse().ToArray();
    }
}
// 2096. Step-By-Step Directions From a Binary Tree Node to Another
/*Solution: Lowest common ancestor

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
Space complexity: O(n)*/
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public string GetDirections(TreeNode root, int startValue, int destValue) {
        StringBuilder startPath = new StringBuilder();
        StringBuilder destPath = new StringBuilder();
        
        buildPath(root, startValue, startPath);
        buildPath(root, destValue, destPath); 

    // Remove common suffix (shared path from root to LCA)
    while (startPath.Length != 0 && destPath.Length != 0
           && startPath[startPath.Length-1] == destPath[destPath.Length-1]) {
        startPath.Remove(startPath.Length - 1, 1);
        destPath.Remove(destPath.Length - 1, 1);
    }
   
    return new string('U',startPath.Length) + new string(destPath.ToString().Reverse().ToArray());
  }
private bool buildPath(TreeNode root, int t, StringBuilder path) {
    if (root == null) return false;
    if (root.val == t) return true;
    if (buildPath(root.left, t, path)) {
      path.Append("L"); 
      return true;
    } else if (buildPath(root.right, t, path)) {
      path.Append("R"); 
      return true;
    }
    return false;
    }
}

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public string GetDirections(TreeNode r, int sV, int dV) {
         StringBuilder sp = new StringBuilder();
        StringBuilder dp = new StringBuilder();
        GetPath(r, sV, sp);
        GetPath(r, dV, dp);
        
        sp = new StringBuilder(new string(sp.ToString().Reverse().ToArray()));
        dp = new StringBuilder(new string(dp.ToString().Reverse().ToArray()));
        
        while (sp.Length > 0 && dp.Length > 0 && sp[0] == dp[0]) {
            sp.Remove(0, 1);
            dp.Remove(0, 1);
        }
        
        for (int i = 0; i < sp.Length; ++i) {
            if (sp[i] == 'L' || sp[i] == 'R') sp[i] = 'U';
        }
        
        return sp.ToString() + dp.ToString();
    }
    
    public bool GetPath(TreeNode r, int v, StringBuilder p) {
        if (r == null) {
            return false;
        } 
        else if (r.val == v) {
            return true;
        } else if (r.left != null && GetPath(r.left, v, p)) {
            p.Append("L");
            return true;
        } else if (r.right != null && GetPath(r.right, v, p)) {
            p.Append("R");
            return true;
        } else {
            return false;
        }
    }
}

// 2095. Delete the Middle Node of a Linked List
/*Solution: Fast / Slow pointers
Use fast / slow pointers to find the previous node of the middle one, 
then skip the middle one.

prev.next = prev.next.next

Time complexity: O(n)
Space complexity: O(1)*/
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
public class Solution {
    public ListNode DeleteMiddle(ListNode head) {
        ListNode dummy = new ListNode(0, head);
    ListNode prev = dummy;
    ListNode fast = head;
    // prev points to the previous node of the middle one.
    while (fast != null && fast.next != null) {
      prev = prev.next;
      fast = fast.next.next;
    }    
    prev.next = prev.next.next;
    return dummy.next;
    }
}

// 2094. Finding 3-Digit Even Numbers
/*Solution: Enumerate all three digits even numbers
Check 100, 102, … 998. Use a hashtable to check whether 
all digits are covered by the given digits.

Time complexity: O(1000*lg(1000))
Space complexity: O(10)
*/
public class Solution {
    public int[] FindEvenNumbers(int[] digits) {
        int[] counts = new int[10]; Array.Fill(counts,0);
    foreach (int d in digits) ++counts[d];
    List<int> ans = new List<int>();
    for (int x = 100; x < 1000; x += 2) {
      bool valid = true;
     int[] c = new int[10]; Array.Fill(c,0);
      for (int t = x; t > 0; t /= 10)
        valid &= (++c[t % 10] <= counts[t % 10]);           
      if (valid) ans.Add(x);
    }
    return ans.ToArray();
        
    }
}

// 123. Best Time to Buy and Sell Stock III
public class Solution {
    public int MaxProfit(int[] prices) {
        int k = 2;
    int n = prices.Length;     
    int[] balance = new int[k + 1]; Array.Fill(balance,Int32.MinValue);
    int[] profit = new int[k + 1];Array.Fill(profit, 0);    
    for (int i = 0; i < n; ++i)
      for (int j = 1; j <= k; ++j) {
        balance[j] = Math.Max(balance[j], profit[j - 1] - prices[i]);
        profit[j] = Math.Max(profit[j], balance[j] + prices[i]);
      }
    return profit[k];
    }
}

// 179. Largest Number
/*Solution: Greedy
Sort numbers by lexical order.
e.g. 9 > 666

Time complexity: O(nlogn)
Space complexity: O(n)*/
public class Solution {
    public string LargestNumber(int[] nums) {
         List<string> s = new List<string>();
    foreach (int x in nums)
      s.Add(x.ToString());
    s.Sort((a, b) => (b + a).CompareTo(a + b));
    if (s[0] == "0") return "0";
    string ans = String.Empty;
    for (int i = 0; i < s.Count; ++i)
      ans += s[i];
    return ans;
    }
}

public class Solution {
    public string LargestNumber(int[] nums) {
         Array.Sort(nums, (a, b) => (b + "" + a).CompareTo(a + "" + b));
        return nums[0] == 0 ? "0" : string.Join("", nums);
    }
}

public class Solution {
    public string LargestNumber(int[] nums) {
         Array.Sort(nums, (a, b) => (b.ToString() + a.ToString()).CompareTo(a.ToString() + b.ToString()));
    return nums[0] == 0 ? "0" : string.Join("", nums);
    }
}

public class Solution {
    public string LargestNumber(int[] nums) {        
        Array.Sort(nums, (a,b) =>
                   {
                      return (b.ToString() + a.ToString()).CompareTo(a.ToString() + b.ToString()); 
                   });
    return nums[0] == 0 ? "0" : string.Join("", nums);
    }
}

public class Solution {
    public string LargestNumber(int[] nums) {
        Array.Sort(nums, (int s1, int s2) =>
            {
                string str1 = s1.ToString() + s2.ToString();
                string str2 = s2.ToString() + s1.ToString();
                return str2.CompareTo(str1);
            });
    return nums[0] == 0 ? "0" : string.Join("", nums);
    }
}

public class Solution {
    public string LargestNumber(int[] nums) {
         var strs = new List<string>(nums.Select(n => n.ToString()));
            strs.Sort((a, b) => (b + a).CompareTo(a + b));
            return strs[0][0] == '0' ? "0" : string.Join("", strs);
    }
}

// 188. Best Time to Buy and Sell Stock IV
/*Solution: Bit Operation
Use n & 1 to get the lowest bit of n.
Use n >>= 1 to right shift n for 1 bit, e.g. removing the last bit.

Time complexity: O(logn)
Space complexity: O(1)*/
public class Solution {
    public int MaxProfit(int k, int[] prices) {
         int n = prices.Length;     
    int[] balance = new int[k + 1]; Array.Fill(balance,Int32.MinValue);
    int[] profit = new int[k + 1];Array.Fill(profit, 0);    
    for (int i = 0; i < n; ++i)
      for (int j = 1; j <= k; ++j) {
        balance[j] = Math.Max(balance[j], profit[j - 1] - prices[i]);
        profit[j] = Math.Max(profit[j], balance[j] + prices[i]);
      }
    return profit[k];
    }
}

// 171. Excel Sheet Column Number
/*Solution: Base conversion
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int TitleToNumber(string columnTitle) {
        int ans = 0;    
    for (int i = columnTitle.Length - 1, cur = 0; i >= 0; --i) {
      cur = cur > 0 ? cur * 26 : 1;
      ans += (columnTitle[i] - 'A' + 1) * cur;
    }
    return ans;
    }
}

public class Solution {
    public int TitleToNumber(string columnTitle) {
        int result = 0;
    for(int i = 0 ; i < columnTitle.Length; i++) {
      result = result * 26 + (columnTitle[i] - 'A' + 1);
    }
    return result;
    }
}

// 168. Excel Sheet Column Title
/*Solution: Base conversion
Time complexity: O(logn)
Space complexity: O(logn)*/
public class Solution {
    public string ConvertToTitle(int columnNumber) {
         string ans = "";
    while (columnNumber > 0){
      columnNumber--;
      ans += (char)('A' + columnNumber % 26); 
      columnNumber /= 26;
    } 
    var c = ans.ToCharArray();
    Array.Reverse(c);
      
	return new String(c);
       
    }
}

public class Solution {
    public string ConvertToTitle(int columnNumber) {
         string ans = "";
    do{
      columnNumber--;
      ans += (char)('A' + columnNumber % 26); 
      columnNumber /= 26;
    } while (columnNumber > 0);
    var c = ans.ToCharArray();
    Array.Reverse(c);
      
	return new String(c);
       
    }
}

public class Solution {
    public string ConvertToTitle(int n) {
         string s = "";
		while (n >= 1) {
			char ch;
			int r = n % 26;
			if (r == 0) {
			  ch = 'Z';  
			  n--;  
			} 
			else {
			  ch = (char) (r + 64);  
			} 
			s = ch + s;
			n = n/ 26;
		}        

		return s;
    }
}

public class Solution {
    public string ConvertToTitle(int columnNumber) {
        Stack<char> stack = new Stack<char>();
        StringBuilder res = new StringBuilder();
        
        while (columnNumber != 0)
        {
            columnNumber--;
            
            stack.Push((char)(columnNumber % 26 + 'A'));
            columnNumber /= 26;
        }
        
        while (stack.Count > 0)
        {
            res.Append(stack.Pop());
        }
        
        return res.ToString();
    }
}

public class Solution {
    public string ConvertToTitle(int columnNumber) {
        StringBuilder sb = new StringBuilder();
        while(columnNumber > 0)
        {    
            columnNumber--;
            char c = (char)(columnNumber % 26 + 'A');
            sb.Append(c.ToString());
            columnNumber /= 26;
        }
        
        char[] arr = sb.ToString().ToCharArray();
        Array.Reverse(arr);
        return new string(arr);
    }
}
// 166. Fraction to Recurring Decimal
/*Solution: Hashtable
Time complexity: O(?)*/
public class Solution {
    public string FractionToDecimal(int numerator, int denominator) {
       
        StringBuilder ss = new StringBuilder(), ss2 = new StringBuilder();
        
        long n = numerator;
        long d = denominator;
 
    if ( n > 0 && d < 0 || n < 0 && d > 0) {
           ss.Append('-');
          n = Math.Abs((long)numerator);
          d = Math.Abs((long)denominator);
    }

       
        ss.Append(n / d); 
        long r = n % d;
 
        bool loop = false;
        int count = 0;
        int loop_start = 0;
        
        if (r != 0)
        {
            n = r;
            ss.Append('.');
            Dictionary<long, int> rs = new Dictionary<long, int>();         
            rs[r] = 0;
        
            do
            {
                n = n*10;
                r = n % d;
                ss2.Append(n / d); 
                n = r;
                if (r != 0 && rs.ContainsKey(r))
                {
                     loop = true;
                    loop_start = rs[r];
                    break;
                }
                rs[r] = ++count;
                
                
            } while(r != 0);

            if (loop)
            {
                string s2 = ss2.ToString();
                ss.Append(s2.Substring(0, loop_start)+"("+ s2.Substring(loop_start) + ")");
        
            }
            else{
                ss.Append(ss2.ToString());
            }
        }
        
        return ss.ToString();
    }
}
// Dictionary and StringBuilder
public class Solution {
    public string FractionToDecimal(int numerator, int denominator) {
        if (numerator == 0)
        {
            return "0";
        }
        
        var result = new StringBuilder();

        if (numerator > 0 ^ denominator > 0)
        {
            result.Append("-");
        }
        
        long n = Math.Abs((long)numerator);
        long d = Math.Abs((long)denominator);
        result.Append(n / d); 
        n %= d;
        
        if (n != 0)
        {
            result.Append('.');
            
            var numerators = new Dictionary<long, int>();
            var decimalIndex = result.Length;
            var repeatIndex = -1;
            var currentIndex = 0;
        
            do
            {
                n *= 10;
                if (numerators.TryGetValue(n, out var index))
                {
                    repeatIndex = index;
                    break;
                }
                
                numerators.Add(n, currentIndex++);
                result.Append(n / d);
                n %= d;
            } while(n != 0);

            if (repeatIndex >= 0)
            {
                result.Insert(repeatIndex + decimalIndex, '(');
                result.Append(')');
            }
        }
        
        return result.ToString();
    }
}

public class Solution {
    public string FractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) {
            return "0";
        }
        StringBuilder res = new StringBuilder();
        // "+" or "-"
        res.Append(((numerator > 0) ^ (denominator > 0)) ? "-" : "");
        long num = Math.Abs((long)numerator);
        long den = Math.Abs((long)denominator);
        
        // integral part
        res.Append(num / den);
        num %= den;
        if (num == 0) {
            return res.ToString();
        }
        
        // fractional part
        res.Append(".");
        Dictionary<long, int> map = new Dictionary<long, int>();
        map[num] = res.Length;
        while (num != 0) {
            num *= 10;
            res.Append(num / den);
            num %= den;
            if (map.ContainsKey(num)) {
                int index = map[num];
                res.Insert(index, "(");
                res.Append(")");
                break;
            }
            else {
                map[num] = res.Length;
            }
        }
        return res.ToString();
    }
}
// 151. Reverse Words in a String
/*Solution: Stack
Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public string ReverseWords(string s) {
        string ss = new string(s);
    
        string w = String.Empty;
        string[] arr = s.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        Stack<string> st = new Stack<string>();
        
        foreach(string str in arr)
        {
            st.Push(str);
        }
        string ans = String.Empty;
        while (st.Count != 0) {
        ans += st.Peek();
        st.Pop();
        if (st.Count != 0) ans += ' ';
        }
        return ans;
    }
}

public class Solution {
    public string ReverseWords(string s) {
        string[] arr = s.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        
        StringBuilder sb = new StringBuilder();
        for(int i = arr.Length - 1; i >= 0; i--)
        {
            sb.Append(arr[i]);
            sb.Append(" ");
        }
        
        return sb.ToString().Trim();
    }
}

public class Solution {
    public string ReverseWords(string s) {
        string[] arr = s.Split(" ", StringSplitOptions.RemoveEmptyEntries);
        
        Array.Reverse(arr);
           
        return String.Join(" ", arr); 
    }
}
/*Approach 4: Split + Stack
Time complexity: O(N)
Space complexity: O(N)*/
public class Solution {
    public string ReverseWords(string s) {
        s = s.Trim().TrimStart();
        
        string[] arr = s.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        Stack<string> stack = new Stack<string>();
        
        foreach(string str in arr)
        {
            stack.Push(str);
        }
        
        StringBuilder sb = new StringBuilder();
        while(stack.Count > 0)
        {
            sb.Append(stack.Pop());
            sb.Append(" ");
        }
        
        return sb.ToString().Trim();
    }
}

public class Solution {
    public string ReverseWords(string s) =>
    String.Join(" ", s.Split(' ',  StringSplitOptions.RemoveEmptyEntries).Reverse());
}
//Linq 
public class Solution {
    public string ReverseWords(string s) {
        //In C# Split(" ") function will turn every " " into ""
		//use Select and where to filter out all ""
		//Aggregate function (x,y)=>y+" "+x will reverse the sequence and insert white space back in.
        return s.Split(" ").Select(x=>x).Where(x=>x!="").Aggregate((x,y)=>y+" "+x);
    }
}

public class Solution {
    public string ReverseWords(string s) {
        string[] arr = s.Trim().Split(" ", StringSplitOptions.RemoveEmptyEntries);
        return String.Join(" ",  Enumerable.Reverse(arr).ToArray()); 
    }
}

public class Solution {
    public string ReverseWords(string s) {
        string[] arr = s.Trim().Split(" ", StringSplitOptions.RemoveEmptyEntries);
         return String.Join(" ",  arr.Reverse()); 
    }
}

public class Solution {
    public string ReverseWords(string s) {
        string[] arr = s.Trim().Split(" ", StringSplitOptions.RemoveEmptyEntries);
        return String.Join(" ", arr.Reverse().ToArray() ); 
    }
}

// 2092. Find All People With Secret
/*Solution: Union Find
Sorting meetings by time.

At each time stamp, union people who meet.
Key step: “un-union” people if they DO NOT connected to 0 / known the secret after each timestamp.

Time complexity: O(nlogn + m + n)
Space complexity: O(m + n)*/
// Array.Sort + Dictionary
public class Solution {
    public IList<int> FindAllPeople(int n, int[][] meetings, int firstPerson) {
        Array.Sort(meetings, (a,b) => a[2]-b[2]);  
        Dictionary<int, List<KeyValuePair<int, int>>> events = new Dictionary<int, List<KeyValuePair<int, int>>>();
    foreach (int[] m in meetings){
         events[m[2]] = events.GetValueOrDefault(m[2], new List<KeyValuePair<int, int>>() );
         events[m[2]].Add(new KeyValuePair<int, int>(m[0], m[1]));
    }
     
    int[] p = Enumerable.Range(0, n).ToArray();
     int find(int x) {
      return p[x] == x ? x : (p[x] = find(p[x]));
    };
        
    p[firstPerson] = 0;
    
    foreach (var (t, s) in events) {
      foreach (var (u, v) in s)
        p[find(u)] = find(v);
      foreach (var (u, v) in s) {
        if (find(u) != find(0)) p[u] = u;
        if (find(v) != find(0)) p[v] = v;
      }
    }    
        
     IList<int> ans = new List<int>();
    for (int i = 0; i < n; ++i)
      if (find(i) == find(0)) ans.Add(i);
    return ans;
        
    }
}
// SortedDictionary slow
public class Solution {
    public IList<int> FindAllPeople(int n, int[][] meetings, int firstPerson) {
          SortedDictionary<int, List<KeyValuePair<int, int>>> events = new SortedDictionary<int, List<KeyValuePair<int, int>>>();
    foreach (int[] m in meetings){
         events[m[2]] = events.GetValueOrDefault(m[2], new List<KeyValuePair<int, int>>() );
         events[m[2]].Add(new KeyValuePair<int, int>(m[0], m[1]));
    }
     
    int[] p = Enumerable.Range(0, n).ToArray();
     int find(int x) {
      return p[x] == x ? x : (p[x] = find(p[x]));
    };
        
    p[firstPerson] = 0;
    
    foreach (var (t, s) in events) {
      foreach (var (u, v) in s)
        p[find(u)] = find(v);
      foreach (var (u, v) in s) {
        if (find(u) != find(0)) p[u] = u;
        if (find(v) != find(0)) p[v] = v;
      }
    }    
        
     IList<int> ans = new List<int>();
    for (int i = 0; i < n; ++i)
      if (find(i) == find(0)) ans.Add(i);
    return ans;
        
    }
}
// BFS
public class Solution {
    public IList<int> FindAllPeople(int n, int[][] meetings, int firstPerson) {
        var map = new SortedDictionary<int, List<int[]>>();
        foreach(var meet in meetings)
        {
            var time = meet[2];
            if(map.ContainsKey(time) == false)
            {
                map.Add(time, new List<int[]>());
            }
            map[time].Add(new int[]{meet[0], meet[1]});
        }
        var res = new HashSet<int>(){0, firstPerson};
        
        var uf = new UnionFind(n);
        uf.Union(0, firstPerson);
        
        foreach(var kvp in map)
        {
            var level = new HashSet<int>();
            foreach(var pair in kvp.Value)
            {
                level.Add(pair[0]);
                level.Add(pair[1]);
                uf.Union(pair[0], pair[1]);
            }
            foreach(var p in level)
            {
                if(uf.Find(p) == 0) res.Add(p);
                else uf.RestoreParents(p);
            }
        }
        return res.ToList();
    }
    public class UnionFind
    {
        private int[] _parents;
        private int[] _rank;
        
        public UnionFind(int n)
        {
            _parents = new int[n];
            _rank = new int[n];
            for(int i = 0; i < n; i++)
            {
                _parents[i] = i;
                _rank[i] = 1;
            }
            _rank[0] = n;
        }
        public void Union(int x, int y)
        {
            int xHead = Find(x);
            int yHead = Find(y);
            if(xHead != yHead)
            {
                int xRank = _rank[xHead];
                int yRank = _rank[yHead];
                if(xRank > yRank) _parents[yHead] = xHead;
                else if(xRank < yRank) _parents[xHead] = yHead;
                else
                {
                    _parents[yHead] = xHead;
                    _rank[xHead]++;
                }
            }
        }
        public int Find(int x)
        {
            if(_parents[x] == x) return x;
            return _parents[x] = Find(_parents[x]);
        }
        private bool IsConnected(int x, int y)
        {
            return Find(x) == Find(y);
        }
        public void RestoreParents(int x)
        {
            _parents[x] = x;
        }
    }
}

// 2091. Removing Minimum and Maximum From Array
/*Solution: Three ways
There are only three ways to remove min/max elements.
1) Remove front elements
2) Remove back elements
3) Remove one with front elements, and another one with back elements.

Just find the best way to do it.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MinimumDeletions(int[] nums) {
        int n = nums.Length;
   // var (it1, it2) = (nums.Min(), nums.Max());
    var (it1, it2) = (nums.Min(), nums.Max());
    int i = Array.IndexOf(nums, it1) ;
    int j = Array.IndexOf(nums, it2) ;
    return Math.Min(Math.Min(Math.Max(i, j) + 1, // remove front elements
                n - Math.Min(i, j)), // remove back elements
                Math.Min(i, j) + 1 + n - Math.Max(i, j) // front + back
                     );    
    }
}
/*Explanation
Find index i of the minimum
Find index j of the maximum

To remove element A[i],
we can remove i + 1 elements from front,
or we can remove n - i elements from back.


Complexity
Time O(n)
Space O(1)*/
public class Solution {
    public int MinimumDeletions(int[] A) {
        int i = 0, j = 0, n = A.Length;
        for (int k = 0; k < n; ++k) {
            if (A[i] < A[k]) i = k;
            if (A[j] > A[k]) j = k;
        }
        return Math.Min(Math.Min(Math.Max(i + 1, j + 1), Math.Max(n - i, n - j)), Math.Min(i + 1 + n - j, j + 1 + n - i));  
    }
}

// 2090. K Radius Subarray Averages
/*Solution: Sliding Window
We compute i – k’s average at position i.

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int[] GetAverages(int[] nums, int k) {
        int n = nums.Length;
    long sum = 0;
    int[] ans = new int[n]; Array.Fill(ans, -1);    
    for (int i = 0; i < n; ++i) {
      sum += nums[i];
      if (i >= 2 * k) {
        ans[i - k] = Convert.ToInt32( sum / (2 * k + 1));           
        sum -= nums[i - 2 * k];
      }
    }
    return ans;
    }
}

// 2089. Find Target Indices After Sorting Array
/*Solution: Sorting
Time complexity: O(nlogn)
Space complexity: O(1)*/
public class Solution {
    public IList<int> TargetIndices(int[] nums, int target) {
        Array.Sort(nums);
    IList<int> ans = new List<int>();
    for (int i = 0; i < nums.Length; ++i)
      if (nums[i] == target) ans.Add(i);
    return ans;
    }
}
// 2088. Count Fertile Pyramids in a Land

// 2087. Minimum Cost Homecoming of a Robot in a Grid
/*Solution: Manhattan distance
Move directly to the goal, no back and forth. 
Cost will be the same no matter which path you choose.

ans = sum(rowCosts[y1+1~y2]) + sum(colCosts[x1+1~x2])

Time complexity: O(m + n)
Space complexity: O(1)*/
public class Solution {
    public int MinCost(int[] startPos, int[] homePos, int[] rowCosts, int[] colCosts) {
    int dx = homePos[1] > startPos[1] ? 1 : (homePos[1] < startPos[1] ? -1 : 0);
    int dy = homePos[0] > startPos[0] ? 1 : (homePos[0] < startPos[0] ? -1 : 0);
    int ans = 0;
    while (homePos[1] != startPos[1]) ans += colCosts[startPos[1] += dx];
    while (homePos[0] != startPos[0]) ans += rowCosts[startPos[0] += dy];    
    return ans;
    }
}

// 2086. Minimum Number of Buckets Required to Collect Rainwater from Houses
/*Solution: Greedy
Try to put a bucket after a house if possible, otherwise put it before the house, or impossible.

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int MinimumBuckets(string street) {
        int n = street.Length;
    int ans = 0;
        
        StringBuilder sb = new StringBuilder(street);
        //sb[pos] = replacement;
        
    for (int i = 0; i < n; ++i){
        
        if (sb[i] == 'H'){
           
        if ((i - 1) >= 0 && sb[i - 1] == 'B')
          continue;
        else if ((i + 1) < n && sb[i + 1] == '.'){
            sb[i+1] = 'B'; ++ans;
        }     
        else if ((i - 1) >= 0 && sb[i - 1] == '.'){
            sb[i-1] = 'B'; ++ans; 
        }
        else
          return -1;
      }   
    }     
    return ans;
    }
}
/*Explanation
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
Space O(1)

The number of "H" in s equals n - s.replace("H", "").length()
The number of "H.H" in s equals n - s.replace("H.H", " ").length()*/
public class Solution {
    public int MinimumBuckets(string s) {
        return (s.Equals("H") || s.StartsWith("HH") || s.EndsWith("HH") || s.Contains("HHH")) ?
               -1 : s.Replace("H.H", "  ").Length - s.Replace("H", "").Length;
    }
}
// 2085. Count Common Words With One Occurrence
/*Solution: Hashtable
Time complexity: O(n + m)
Space complexity: O(n + m)

*/
public class Solution {
    public int CountWords(string[] words1, string[] words2) {
    Dictionary<string, int> c1 = new Dictionary<string, int>();
    Dictionary<string, int> c2 = new Dictionary<string, int>();
    foreach (string w in words1) c1[w] = c1.GetValueOrDefault(w,0)+1;
    foreach (string w in words2) c2[w] = c2.GetValueOrDefault(w,0)+1;
    int ans = 0;
    foreach (var (w, c) in c1)
      if (c == 1 && c2.ContainsKey(w) && c2[w] == 1) ++ans;
    return ans;
    }
}

// 137. Single Number II
/*Solution: Bit by bit
Since every number appears three times, the i-th bit must be a factor of 3, if not, that bit belongs to the single number.

Time complexity: O(32n)
Space complexity: O(1)

*/
public class Solution {
    public int SingleNumber(int[] nums) {
        int ans = 0;
    for (int s = 0; s < 32; ++s) {
      int mask = 1 << s;
      int sum = 0;
      foreach (int x in nums)
        if ((x & mask ) != 0) ++sum;
      if ((sum % 3 )!= 0) ans |= mask;
    }
    return ans;
    }
}

// 114. Flatten Binary Tree to Linked List

// 109. Convert Sorted List to Binary Search Tree
/*Solution 1: Recursion w/ Fast + Slow Pointers
For each sublist, use fast/slow pointers to find the mid and build the tree.

Time complexity: O(nlogn)
Space complexity: O(logn)

*/
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public TreeNode SortedListToBST(ListNode head) {
        TreeNode build(ListNode head, ListNode tail){
      if (head == null || head == tail) return null;
      ListNode fast = head, slow = head;
      while (fast != tail && fast.next != tail) {
        slow = slow.next;
        fast = fast.next.next;
      }
      TreeNode root = new TreeNode(slow.val);
      root.left = build(head, slow);
      root.right = build(slow.next, tail);
      return root;
    };
    return build(head, null);
    }
}

// 41. First Missing Positive
/*Solution: Marking
First pass, marking nums[i] to INT_MAX if nums[i] <= 0
Second pass, use a negative number to mark the presence of a number x at nums[x – 1]
Third pass, the first positive number is the missing index i, return i +1
If not found return n + 1.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int FirstMissingPositive(int[] nums) {
        int n = nums.Length;
    for (int i = 0; i < n; i++)
      if (nums[i] <= 0) nums[i] = Int32.MaxValue;
    
   for (int i = 0; i < n; i++) {
      int num = Math.Abs(nums[i]);
      if (num >= 1 && num <= n && nums[num - 1] > 0)
        nums[num - 1] *= -1;
    }
    
    for (int i = 0; i < n; ++i)
      if (nums[i] > 0)
        return i + 1;
    
    return n + 1;
    }
}
/*Numbers greater then n can be ignored because the missing integer must be in the range 1..n+1

If each cell in the array were to contain positive integers only, 
we can use the negative of the stored number as a flag to mark something 
(in this case the flag indicates this index was found in some cell of the array)*/
public class Solution {
    public int FirstMissingPositive(int[] nums) {
       int n = nums.Length;
    
    // 1. mark numbers (num < 0) and (num > n) with a special marker number (n+1) 
    // (we can ignore those because if all number are > n then we'll simply return 1)
    for (int i = 0; i < n; i++) {
        if (nums[i] <= 0 || nums[i] > n) {
            nums[i] = n + 1;
        }
    }
    // note: all number in the array are now positive, and on the range 1..n+1
    
    // 2. mark each cell appearing in the array, by converting the index for that number to negative
    for (int i = 0; i < n; i++) {
        int num = Math.Abs(nums[i]);
        if (num > n) {
            continue;
        }
        num--; // -1 for zero index based array (so the number 1 will be at pos 0)
        if (nums[num] > 0) { // prevents double negative operations
            nums[num] = -1 * nums[num];
        }
    }
    
    // 3. find the first cell which isn't negative (doesn't appear in the array)
    for (int i = 0; i < n; i++) {
        if (nums[i] >= 0) {
            return i + 1;
        }
    }
    
    // 4. no positive numbers were found, which means the array contains all numbers 1..n
    return n + 1;
    }
}

// 30. Substring with Concatenation of All Words
/*Solution: Hashtable + Brute Force
Try every index and use a hashtable to check coverage.

Time complexity: O(n*m*l)
Space complexity: O(m*l)
*/
public class Solution {
    public IList<int> FindSubstring(string s, string[] words) {
        int n = s.Length;
    int m = words.Length;
    int l = words[0].Length;
    if (m * l > n) return new List<int>();
    Dictionary<string, int> freq = new Dictionary<string, int>();
    foreach (string word in words) freq[word] = freq.GetValueOrDefault(word,0)+1;
    
    IList<int> ans = new List<int>();
    for (int i = 0; i <= n - m * l; ++i) {
      Dictionary<string, int> seen = new Dictionary<string, int>();
      int count = 0;
      for (int k = 0; k < m; ++k) {
        string t = s.Substring(i + k * l, l);
        seen[t]  = seen.GetValueOrDefault(t,0)+1 ;
        if ( seen[t] > freq.GetValueOrDefault(t,0)) break;

        ++count;  
      }
      if (count == m) ans.Add(i);
    }
    return ans;
    }
}

public class Solution {
    public IList<int> FindSubstring(string s, string[] words) {
        int n = s.Length;
    int m = words.Length;
    int l = words[0].Length;
    if (m * l > n) return new List<int>();
    Dictionary<string, int> freq = new Dictionary<string, int>();
    foreach (string word in words) freq[word] = freq.GetValueOrDefault(word,0)+1;
    
    IList<int> ans = new List<int>();
    for (int i = 0; i <= n - m * l; ++i) {
      Dictionary<string, int> seen = new Dictionary<string, int>();
      int count = 0;
      for (int k = 0; k < m; ++k) {
        string t = s.Substring(i + k * l, l);
        freq[t] = freq.GetValueOrDefault(t,0); 
         seen[t]  = seen.GetValueOrDefault(t,0)+1 ;
        if ( seen[t] > freq[t]) break;

        ++count;  
      }
      if (count == m) ans.Add(i);
    }
    return ans;
    }
}

// 2007. Find Original Array From Doubled Array
/*Solution 2: Hashtable
Time complexity: O(max(nums) + n)
Space complexity: O(max(nums))
*/
public class Solution {
    public int[] FindOriginalArray(int[] changed) {
          if ((changed.Length & 1) != 0) return new int[0];
    Array.Sort(changed);
    int n = changed.Length / 2;
    int kMax = changed.Max();
    int[] m = new int[kMax + 1]; //Array.Fill(m,0);
    foreach (int x in changed) ++m[x];
    if ((m[0] & 1) != 0 ) return new int[0];
    List<int> ans = Enumerable.Repeat(default(int), m[0] / 2).ToList();//new int[m[0] / 2];Array.Fill(ans, 0);
    for (int x = 1; ans.Count != n; ++x) {
      if (x * 2 > kMax || m[x * 2] < m[x]) return new int[0];
        ans.AddRange(Enumerable.Repeat(x, m[x]));
      //ans.Add(end(ans), m[x], x);
      m[x * 2] -= m[x];   
    }
    return ans.ToArray();
    }
}

public class Solution {
    public int[] FindOriginalArray(int[] changed) {
          if ((changed.Length & 1) != 0) return new int[0];
    Array.Sort(changed);
    int n = changed.Length / 2;
    int kMax = changed.Max();
    int[] m = new int[kMax + 1]; //Array.Fill(m,0);
    foreach (int x in changed) ++m[x];
    if ((m[0] & 1) != 0 ) return new int[0];
    List<int> ans = Enumerable.Repeat(default(int), m[0] / 2).ToList();//new int[m[0] / 2];Array.Fill(ans, 0);
    for (int x = 1; ans.Count != n; ++x) {
      if (x * 2 > kMax || m[x * 2] < m[x]) return new int[0];
        for (int j = 0; j < m[x]; ++j) { ans.Add(x); }
      //ans.Add(end(ans), m[x], x);
      m[x * 2] -= m[x];   
    }
    return ans.ToArray();
    }
}

public class Solution {
    public int[] FindOriginalArray(int[] changed) {
          if ((changed.Length & 1) != 0) return new int[0];
    Array.Sort(changed);
    int n = changed.Length / 2;
    int kMax = changed.Max();
    int[] m = new int[kMax + 1]; //Array.Fill(m,0);
    foreach (int x in changed) ++m[x];
    if ((m[0] & 1) != 0 ) return new int[0];
    List<int> ans = new List<int>(new int[m[0]/2]);//new int[m[0] / 2];Array.Fill(ans, 0);
    for (int x = 1; ans.Count != n; ++x) {
      if (x * 2 > kMax || m[x * 2] < m[x]) return new int[0];
        for (int j = 0; j < m[x]; ++j) {
                ans.Add(x);}
      //ans.Add(end(ans), m[x], x);
      m[x * 2] -= m[x];   
    }
    return ans.ToArray();
    }
}
/*Match from the Smallest or Biggest

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
And we still need to check all other numbers.*/
public class Solution {
    public int[] FindOriginalArray(int[] A) {
          int n = A.Length, i = 0;
        Array.Sort(A);
        if (n % 2 == 1) return new int[0];
        int[] res = new int[n / 2];
        Dictionary<int, int> count = new Dictionary<int, int>();
        foreach (int a in A)
            count[a] = count.GetValueOrDefault(a, 0) + 1;
        //List<int> c = count.Keys.OrderBy(x => x).ToList();
        foreach (int x in count.Keys) {
            if (count[x] > count.GetValueOrDefault(x + x, 0))
                return new int[0];
            for (int j = 0; j < count[x]; ++j) {
                res[i++] = x;
                count[x + x] = count.GetValueOrDefault(x + x, 0) - 1;
            }
        }
        return res;
    }
}

public class Solution {
    public int[] FindOriginalArray(int[] A) {
          int n = A.Length, i = 0;
        if (n % 2 == 1) return new int[0];
        int[] res = new int[n / 2];
        Dictionary<int, int> count = new Dictionary<int, int>();
        foreach (int a in A)
            count[a] = count.GetValueOrDefault(a, 0) + 1;
        List<int> c = count.Keys.OrderBy(x => x).ToList();
        foreach (int x in c) {
           // count[x + x] = count.GetValueOrDefault(x + x, 0);
            if (count[x] > count.GetValueOrDefault(x + x, 0))
                return new int[0];
            for (int j = 0; j < count[x]; ++j) {
                res[i++] = x;
                count[x + x] = count.GetValueOrDefault(x + x, 0) - 1;
            }
        }
        return res;
    }
}
// 2008. Maximum Earnings From Taxi
/*DP solution

Explanation
Sort A, solve it like Knapsack dp.


Complexity
Time O(n + klogk), k = A.length
Space O(n)*/
public class Solution {
    public long MaxTaxiEarnings(int n, int[][] A) {
        Array.Sort(A, (a, b) => a[0] - b[0]);
         
        long[] dp = new long[n + 1];
        int j = 0;
        for(int i = 1; i <= n; ++i) {
            dp[i] = Math.Max(dp[i], dp[i - 1]);
            while (j < A.Length && A[j][0] == i) {
                dp[A[j][1]] = Math.Max(dp[A[j][1]], dp[i] + A[j][1] - A[j][0] + A[j][2]);
                ++j;
            }
        }
        return dp[n];
    }
}
/*Solution: DP
dp[i] := max earnings we can get at position i and the taxi is empty.

dp[i] = max(dp[i – 1], dp[s] + gain) where e = i, gain = e – s + tips

For each i, we check all the rides that end at i and find the best one (which may have different starting points), otherwise the earning will be the same as previous position (i – 1).

answer = dp[n]

Time complexity: O(m + n)
Space complexity: O(m + n)*/
public class Solution {
    public long MaxTaxiEarnings(int n, int[][] rides) {
        long[] dp = new long[n + 1];//Array.Fill(dp,0);
        Dictionary<int, List<KeyValuePair<int, int>>> m = new Dictionary<int, List<KeyValuePair<int, int>>>();
        
        foreach (int[] r in rides){
            m[r[1]] = m.GetValueOrDefault(r[1], new List<KeyValuePair<int, int>>());
        m[r[1]].Add(new KeyValuePair<int, int>(r[0], r[1] - r[0] + r[2]));
        }
        for (int i = 1; i <= n; ++i) {
        dp[i] = dp[i - 1];
            if(m.ContainsKey(i)){
                foreach (var (s, g) in m[i])
                    dp[i] = Math.Max(dp[i], g + dp[s]);
            }
        }
        return dp[n];
    }
}
// 2009. Minimum Number of Operations to Make Array Continuous
/*Solution: Sliding Window
Remove duplicates and sort the numbers.
Try using nums[i] as the min number of the final array.
window [i, j), max – min < n, then change the rest of array to fit into or append after the window, which takes n – (j – i) steps.
e.g. input = [10, 3, 1, 4, 5, 6, 6, 6, 11, 15] => sorted + unique => [1, 3, 4, 5, 6, 10, 11, 15]
n = 10, window = [3, 4, 5, 6, 10, 11], max = 11, min = 3, max – min = 8 < 10
Final array = [3, 4, 5, 6, 1->7, 62->8, 63->9, 10, 11, 15->12]
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MinOperations(int[] A) {
        int n = A.Length;
    Array.Sort(A);
    A = A.Distinct().ToArray();
    //A.erase(unique(begin(A), end(A)), end(A));    
    int ans = Int32.MaxValue;
    for (int i = 0, j = 0, m = A.Length; i < m; ++i) {
      while (j < m && A[j] < A[i] + n) ++j;
      ans = Math.Min(ans, n - (j - i));
    }
    return ans;
    }
}

// 2006. Count Number of Pairs With Absolute Difference K
/*Solution: Hashtable
|y – x| = k
y = x + k or y = x – k
Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int CountKDifference(int[] nums, int k) {        
    Dictionary<int,int> m = new Dictionary<int,int>();
        int ans = 0;
        
        foreach(int x in nums){
            ans += m.GetValueOrDefault(x-k,0);
            ans += m.GetValueOrDefault(x+k,0);           
            m[x] = m.GetValueOrDefault(x,0)+1;
        }
        return ans;
    }
}

public class Solution {
    public int CountKDifference(int[] nums, int k) {        
    Dictionary<int,int> m = new Dictionary<int,int>();
        int ans = 0;
        foreach(int x in nums){
            if(m.ContainsKey(x - k)){
                ans += m[x - k];
            }
            if(m.ContainsKey(x + k)){
                ans += m[x + k];
            }
            m[x] = m.GetValueOrDefault(x,0)+1;
        }
        return ans;
    }
}

// 2081. Sum of k-Mirror Numbers

// 2080. Range Frequency Queries
/*Solution: Hashtable + Binary Search

Time complexity: Init: O(max(arr) + n), query: O(logn)
Space complexity: O(max(arr) + n)*/
public class RangeFreqQuery {

    private Dictionary<int, List<int>> count = new Dictionary<int, List<int>>();
    public RangeFreqQuery(int[] arr) {
        for (int i = 0; i < arr.Length; i++) {
            count[arr[i]] = count.GetValueOrDefault(arr[i], new List<int>());
             count[arr[i]].Add(i);
        }
    }
    
    public int Query(int left, int right, int v) {
        List<int> m = count.GetValueOrDefault(v, new List<int>());
    if (m.Count == 0 ) return 0;
    int r = upperBound(m, right) - 1;
    int l = lowerBound(m, left);    
    return r - l + 1;
    }
    
    public int lowerBound(List<int> nums, int target ) {
      int l = 0, r = nums.Count - 1;

      while (l <= r) {
            int m = (l + r) / 2;
        if (nums[m] >= target) {
          r = m - 1;
        } else {
          l = m + 1;
        }
      }

      return l;
    }

  public int upperBound(List<int> nums, int target ) {
    int l = 0, r = nums.Count - 1;

    while( l <= r ){
          int m = (l + r) / 2;
      if (nums[m] > target) {
        r = m - 1;
      } else {
        l = m + 1;
      }
    }

    return l;
  } 
  }
/**
 * Your RangeFreqQuery object will be instantiated and called as such:
 * RangeFreqQuery obj = new RangeFreqQuery(arr);
 * int param_1 = obj.Query(left,right,value);
 */

 public class RangeFreqQuery {

    private Dictionary<int, List<int>> count = new Dictionary<int, List<int>>();
    public RangeFreqQuery(int[] arr) {
        for (int i = 0; i < arr.Length; i++) {
            count[arr[i]] = count.GetValueOrDefault(arr[i], new List<int>());
             count[arr[i]].Add(i);
        }
    }
    
    public int Query(int left, int right, int v) {
        List<int> m = count.GetValueOrDefault(v, new List<int>());
    if (m.Count == 0 ) return 0;
   // int r = upperBound(m, right) - 1;
        int r = m.BinarySearch(right+1); // directly use C# binary search find upperbound, need to r - 1 later!
            if (r < 0) { r = ~r;}
   // int l = lowerBound(m, left); 
         int l = m.BinarySearch(left); // directly use C# binary search find lowerbound
            if (l < 0) { l = ~l;}
    return r - 1 - l + 1;
    }
  }
/**
 * Your RangeFreqQuery object will be instantiated and called as such:
 * RangeFreqQuery obj = new RangeFreqQuery(arr);
 * int param_1 = obj.Query(left,right,value);
 */

public class RangeFreqQuery {

    private Dictionary<int, List<int>> count = new Dictionary<int, List<int>>();
    public RangeFreqQuery(int[] arr) {
        for (int i = 0; i < arr.Length; i++) {
            count[arr[i]] = count.GetValueOrDefault(arr[i], new List<int>());
             count[arr[i]].Add(i);
        }
    }
    
    public int Query(int left, int right, int v) {
        List<int> m = count.GetValueOrDefault(v, new List<int>());
        if (m.Count == 0 ) return 0;
   // int r = upperBound(m, right) - 1;
        int r = m.BinarySearch(right+1); // directly use C# binary search here
        if (r < 0) { r = ~r;}
        r -= 1;
   // int l = lowerBound(m, left); 
        int l = m.BinarySearch(left); // directly use C# binary search here
        if (l < 0) { l = ~l;}
        return r - l + 1;
    }
  }
/**
 * Your RangeFreqQuery object will be instantiated and called as such:
 * RangeFreqQuery obj = new RangeFreqQuery(arr);
 * int param_1 = obj.Query(left,right,value);
 */

// 2079. Watering Plants
/*Solution: Simulation
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int WateringPlants(int[] plants, int capacity) {
        int n = plants.Length;
    int ans = 0;
    for (int i = 0, c = capacity; i < n; c -= plants[i++], ++ans) {
      if (c >= plants[i]) continue;
      ans += i * 2;
      c = capacity;      
    }
    return ans;
    }
}

// 2078. Two Furthest Houses With Different Colors
/*Solution 1: Brute Force
Try all pairs.
Time complexity: O(n2)
Space complexity: O(1)*/
public class Solution {
    public int MaxDistance(int[] colors) {
        int n = colors.Length;
    for (int d = n - 1; d > 0; --d)
      for (int i = 0; i + d < n; ++i)
        if (colors[i] != colors[i + d])
          return d;
    return 0;
    }
}
/*Solution 2: Greedy / One pass
First house or last house must be involved in the ans.
Scan the house and check with first and last house.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MaxDistance(int[] colors) {
        int ans = 0;
        int n = colors.Length;
    for (int i = 0; i < n; ++i)
      ans = Math.Max( Math.Max(ans, 
                 i * Convert.ToInt32(colors[i] != colors[0])),
                 (n - i - 1) * Convert.ToInt32(colors[i] != colors[n - 1]));
    return ans;
    }
}

// 2050. Parallel Courses III
/*Solution: Topological Sorting
Time complexity: O(V+E)
Space complexity: O(V+E)
*/
public class Solution {
    public int MinimumTime(int n, int[][] relations, int[] time) {
        List<int>[] g = new List<int>[n];  
    foreach (int[] r in relations){
        if (g[r[0]-1] == null) g[r[0] - 1] = new List<int>();
        g[r[0] - 1].Add(r[1] - 1); 
    }
         
    int[] t = new int[n]; Array.Fill(t, -1);
    int dfs(int u) {
    if (t[u] != -1) return t[u];
      t[u] = 0;
    if (g[u] != null) {
      foreach (int v in g[u]){
           t[u] = Math.Max(t[u], dfs(v));
      }
    }
     
      return t[u] += time[u];
    };
    int ans = 0;
    for (int i = 0; i < n; ++i)
      ans = Math.Max(ans, dfs(i));
    return ans;
    }
}

// 2003. Smallest Missing Genetic Value in Each Subtree
/*Solution: DFS on a single path
One ancestors of node with value of 1 will have missing values greater than 1. We do a dfs on the path that from node with value 1 to the root.

Time complexity: O(n + max(nums))
Space complexity: O(n + max(nums))

*/
public class Solution {
    public int[] SmallestMissingValueSubtree(int[] parents, int[] nums) {
         int n = parents.Length;
    int[] ans = new int[n]; Array.Fill(ans, 1);
    int[] seen = new int[100002]; //Array.Fill(seen, 0);
    List<int>[] g = new List<int>[n];
    for (int i = 1; i < n; ++i){
         if (g[parents[i]] == null) g[parents[i]] = new List<int>();
        g[parents[i]].Add(i);
    }
      
    void dfs(int u) {
       if (seen[nums[u]]++ > 0) {return; }
      if (g[u] != null) {
          foreach (int v in g[u]){
          dfs(v);
        } 
      }
      
    };
    int u = Array.IndexOf(nums,1);
    for (int l = 1; u < n && u != -1; u = parents[u]) {
      dfs(u);
      while (seen[l] > 0) ++l;
      ans[u] = l;
    }
    return ans;
    }
}

// 2002. Maximum Product of the Length of Two Palindromic Subsequences
/*Solution 1: DFS
Time complexity: O(3n*n)
Space complexity: O(n)*/
public class Solution {
    public int MaxProduct(string s) {
         bool isPalindrom(StringBuilder s) {
      if(s == null) return false;
      for (int i = 0; i < s.Length; ++i)
        if (s[i] != s[s.Length - i - 1]) return false;
      return true;
    };
    int n = s.Length;
    List<StringBuilder> ss = Enumerable.Repeat((StringBuilder)null, 3).ToList(); //new List<StringBuilder>(3); // new List<string>(3);
    int ans = 0;
    void dfs(int i) {

      if (i == n ) {
          
        if (isPalindrom(ss[0]) && isPalindrom(ss[1]))
          ans = Math.Max(ans, ss[0].Length * ss[1].Length);
        return;
      }
       
   
      for (int k = 0; k < 3; ++k) {
        if (ss[k] == null) ss[k] = new StringBuilder(""); 
        ss[k].Append(s[i]); 
        dfs(i + 1); 
        if (ss[k].Length >= 1) ss[k].Length--;
      }      
    };
    dfs(0);
    return ans;
    }
}
/*Solution: Subsets + Bitmask + All Pairs
Time complexity: O(22n)
Space complexity: O(2n)*/
// use System.Numerics BitOperations.PopCount
using System.Numerics;
public class Solution {
    public int MaxProduct(string s) {
         bool isPalindrom( StringBuilder s) {
              if(s == null) return false;
      for (int i = 0; i < s.Length; ++i)
        if (s[i] != s[s.Length - i - 1]) return false;
      return true;
    };
    int n = s.Length;
    List<int> p = new List<int>();
    for (int i = 0; i < (1 << n); ++i) {
      StringBuilder t = new StringBuilder("");
      for (int j = 0; j < n; ++j)
        if ((i >> j & 1) > 0) t.Append(s[j]);
      if (isPalindrom(t))
        p.Add(i);
    }
    int ans = 0;
    foreach (int s1 in p)
      foreach (int s2 in p) 
        if ((s1 & s2) == 0)
          ans = Math.Max(ans, BitOperations.PopCount((uint)s1) * BitOperations.PopCount((uint)s2));
    return ans;
    }
}
/*Solution: Subsets + Bitmask + All Pairs
Time complexity: O(22n)
Space complexity: O(2n)*/
// use own bit count method
public class Solution {
    public int MaxProduct(string s) {
         bool isPalindrom( StringBuilder s) {
              if(s == null) return false;
      for (int i = 0; i < s.Length; ++i)
        if (s[i] != s[s.Length - i - 1]) return false;
      return true;
    };
    int n = s.Length;
    List<int> p = new List<int>();
    for (int i = 0; i < (1 << n); ++i) {
      StringBuilder t = new StringBuilder("");
      for (int j = 0; j < n; ++j)
        if ((i >> j & 1) > 0) t.Append(s[j]);
      if (isPalindrom(t))
        p.Add(i);
    }
    int ans = 0;
    foreach (int s1 in p)
      foreach (int s2 in p) 
        if ((s1 & s2) == 0)
          ans = Math.Max(ans, count1(s1) * count1(s2));
    return ans;
    }
    
    public int count1(int n) {
    // count number of 1 bits
        int count = 0;
    
        while (n>0) {
            n &= (n - 1);
            count++;
        }

        return count;
    }
}
// 2001. Number of Pairs of Interchangeable Rectangles
/*Solution: Hashtable
Use aspect ratio as the key.

Time complexity: O(n)
Space complexity: O(n)
*/
public class Solution {
    public long InterchangeableRectangles(int[][] rectangles) {
        long ans  = 0;
    Dictionary<long, int> m = new Dictionary<long, int>();
    foreach (int[] r in rectangles) {
      long d = gcd(r[0], r[1]);
      ans +=  m.GetValueOrDefault((((r[0] / d) << 20) | (r[1] / d)), 0);
        m[((r[0] / d) << 20) | (r[1] / d)] = m.GetValueOrDefault((((r[0] / d) << 20) | (r[1] / d)), 0) + 1;
    }
    return ans;
    }
    private int gcd(int a, int b) {
        return b > 0 ? gcd(b, a % b) : a;
    }
}

// 2000. Reverse Prefix of Word
/*Solution: Brute Force
Time complexity: O(n)
Space complexity: O(n) / O(1)*/
public class Solution {
    public string ReversePrefix(string word, char ch) {
        for (int i = 0; i < word.Length; ++i)
            if (word[i] == ch) return new string(word.Substring(0, i + 1).Reverse().ToArray()) + word.Substring(i + 1);
    return word;
    }
}

public class Solution {
    public string ReversePrefix(string word, char ch) {

        if(word.IndexOf(ch) != -1)
        {
            return new string( word.Substring(0, word.IndexOf(ch) + 1).Reverse().ToArray()) + word.Substring(word.IndexOf(ch) + 1);
        }
        return word;
    }
}