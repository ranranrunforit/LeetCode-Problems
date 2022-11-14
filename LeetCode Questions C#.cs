// 2452. Words Within Two Edits of Dictionary
/*Solution: Hamming distance + Brute Force
For each query word q, check the hamming distance 
between it and all words in the dictionary.

Time complexity: O(|q|*|d|*n)
Space complexity: O(1)*/
public class Solution {
    public IList<string> TwoEditWords(string[] queries, string[] dictionary) {
        int n = queries[0].Length;
    bool check(string q) {      
      foreach (string w in dictionary) {
        int dist = 0;
        for (int i = 0; i < n && dist <= 3; ++i)
          dist += Convert.ToInt32(q[i] != w[i]);
        if (dist <= 2) return true;
      }
      return false;
    };
    IList<string> ans = new List<string>();
    foreach ( string q in queries) 
      if (check(q)) ans.Add(q);
    return ans;
    }
}

// 2451. Odd String Difference
/*Solution: Comparing with first string.
Let us pick words[0] as a reference for comparison, 
assuming it’s valid. If we only found one instance say words[i], 
that is different than words[0], we know that words[i] is bad, 
otherwise we should see m – 1 different words which means words[0] itself is bad.

Time complexity: O(m*n)
Space complexity: O(1)*/
public class Solution {
    public string OddString(string[] words) {
        int m = words.Length;
    int n = words[0].Length;
    int count = 0;
    int bad = 0;
    for (int i = 1; i < m; ++i) 
      for (int j = 1; j < n; ++j) {
        if (words[i][j] - words[i][j - 1] 
           != words[0][j] - words[0][j - 1]) {
          ++count;
          bad = i;
          break;
        }
      }
    return words[count == 1 ? bad : 0];
    }
}

// 2441. Largest Positive Integer That Exists With Its Negative
/*Solution 1: Hashtable
We can do in one pass by checking whether -x in the hashtable and 
update ans with abs(x) if so.

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int FindMaxK(int[] nums) {
        HashSet<int> s = new HashSet<int>();
    int ans = -1;
    foreach (int x in nums) {
      if (Math.Abs(x) > ans && s.Contains(-x))
        ans = Math.Abs(x);
      s.Add(x);
    }
    return ans;
    }
}
/*Solution 2: Sorting
Sort the array by abs(x) in descending order.

[-1,10,6,7,-7,1] becomes = [-1, 1, 6, -7, 7, 10]

Check whether arr[i] = -arr[i-1].

Time complexity: O(nlogn)
Space complexity: O(1)*/
public class Solution {
    public int FindMaxK(int[] nums) {
       Array.Sort(nums, (x, y) => Math.Abs(x) == Math.Abs(y) ? x - y :  Math.Abs(y) - Math.Abs(x) );
 
    for (int i = 1; i < nums.Length; ++i)
      if (nums[i] == -nums[i - 1]) return nums[i];
    return -1;
    }
}
/*Solution 3: Two Pointers
Sort the array.

Let sum = nums[i] + nums[j], sum == 0, we find one pair, if sum < 0, ++i else –j.

Time complexity: O(nlogn)
Space complexity: O(1)*/
public class Solution {
    public int FindMaxK(int[] nums) {
        Array.Sort(nums);
    int ans = -1;
    for (int i = 0, j = nums.Length - 1; i < j; ) {
      int s = nums[i] + nums[j];
      if (s == 0) {
        ans = Math.Max(ans, nums[j]);
        ++i; --j;      
      } else if (s < 0) {
        ++i;
      } else {
        --j;
      }      
    }
    return ans;
    }
}
// Linq
public class Solution {
    public int FindMaxK(int[] nums) {
        var x = nums.Where(n => nums.Contains(-n)).OrderByDescending(n => n).ToList();
        if (x.Count == 0)
            return -1;

        return x.First();
    }
}
// 2435. Paths in Matrix Whose Sum Is Divisible by K
/*Let dp[i][j][r] := # of paths from (0,0) to (i,j) with path sum % k == r.

init: dp[0][0][grid[0][0] % k] = 1

dp[i][j][(r + grid[i][j]) % k] = dp[i-1][j][r] + dp[i][j-1][r]

ans = dp[m-1][n-1][0]

Time complexity: O(m*n*k)
Space complexity: O(m*n*k) -> O(n*k)*/
public class Solution {
    public int NumberOfPaths(int[][] grid, int k) {
        int kMod = 1000_000_007;
    int m = grid.Length;
    int n = grid[0].Length;
    int[][][] dp = new int[m][][];
   for (int a = 0; a < m; ++a){
        dp[a] = new int[n][];
        for (int b = 0; b < n; ++b) {
            dp[a][b] =new int[k];}}
    dp[0][0][grid[0][0] % k] = 1;
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j) {
        if (i == 0 && j == 0) continue;
        for (int r = 0; r < k; ++r)
          dp[i][j][(r + grid[i][j]) % k] = 
            ((j > 0 ? dp[i][j - 1][r] : 0) + (i > 0 ? dp[i - 1][j][r] : 0)) % kMod;          
      }
    }
      
    return dp[m - 1][n - 1][0];
    }
}

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
1. movies tracks the {movie -> price} of each shop. 
This is readonly to get the price of a movie for generating keys for treesets.
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

// 2065. Maximum Path Quality of a Graph
/*Solution: DFS
Given time >= 10 and maxTime <= 100, the path length is at most 10, 
given at most four edges connected to each node.
Time complexity: O(410)
Space complexity: O(n)
*/
public class Solution {
    public int MaximalPathQuality(int[] values, int[][] edges, int maxTime) {
        int n = values.Length;
    List<KeyValuePair<int, int>>[] g = new List<KeyValuePair<int, int>>[n];
    foreach (int[] e in edges) {
        if(g[e[0]] == null){g[e[0]] = new List<KeyValuePair<int, int>>();}
        if(g[e[1]] == null){g[e[1]] = new List<KeyValuePair<int, int>>();}
      g[e[0]].Add(new KeyValuePair<int, int>(e[1], e[2]));
      g[e[1]].Add(new KeyValuePair<int, int>(e[0], e[2]));
    }
    int[] seen = new int[n]; Array.Fill(seen,0);
    int ans = 0;
    void dfs(int u, int t, int s) {
      if (++seen[u] == 1) s += values[u];
      if (u == 0) ans = Math.Max(ans, s);
       if (g[u] != null){
        foreach (var (v, d) in g[u])
            if (t + d <= maxTime) dfs(v, t + d, s);
        } 

         if ( --seen[u] == 0 ) s -= values[u];
      
    };
    dfs(0, 0, 0);
    return ans;
    }
}

// 2068. Check Whether Two Strings are Almost Equivalent
/*Solution: Hashtable
Use a hashtable to track the relative frequency of a letter.

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public bool CheckAlmostEquivalent(string word1, string word2) {
        int[] m = new int[26];
    foreach (char c in word1) ++m[c - 'a'];
    foreach (char c in word2) --m[c - 'a'];
    for (int i = 0; i < 26; ++i)
      if (Math.Abs(m[i]) > 3) return false;
    return true;
    }
}

// 2069. Walking Robot Simulation II
/*Solution: Simulation
Note num >> w + h, when we hit a wall, 
we will always follow the boundary afterwards. 
We can do num %= p to reduce steps, where p = ((w – 1) + (h – 1)) * 2

Time complexity: move: O(min(num, w+h))
Space complexity: O(1)*/
public class Robot {

    public static int[][] dirs = new int[4][]{new int[2]{1, 0}, new int[2]{0, 1}, new int[2]{-1, 0}, new int[2]{0, -1}};
    public static string[] kNames = new string[4]{"East", "North", "West", "South"};
 private int w_;
 private int h_;
 private int d_= 0; // 0: E, 1: N, 2: W, 3: S
 private int x_= 0;
 private int y_= 0;
 private int p_= 0;
    public Robot(int width, int height) {
        w_ = width; h_ = height;
        p_ = ((w_ - 1) + (h_ - 1)) * 2;
    }
    
    public void Step(int num) {
         while (num > 0) {
      int tx = x_ + dirs[d_][0];
      int ty = y_ + dirs[d_][1];
      if (tx < 0 || tx >= w_ || ty < 0 || ty >= h_) {
        if (num > p_) num %= p_;
        if (num > 0) d_ = (d_ + 1) % 4;        
        continue;
      }
      x_ = tx;
      y_ = ty;
      --num;
    }
    }
    
    public int[] GetPos() {
        return new int[2]{x_, y_}; 
    }
    
    public string GetDir() {
         return kNames[d_]; 
    }

}

/**
 * Your Robot object will be instantiated and called as such:
 * Robot obj = new Robot(width, height);
 * obj.Step(num);
 * int[] param_2 = obj.GetPos();
 * string param_3 = obj.GetDir();
 */

// 2070. Most Beautiful Item for Each Query
/*Solution: Prefix Max + Binary Search
Sort items by price. 
For each price, use a treemap to store the max beauty of an item 
whose prices is <= p. 
Then use binary search to find the max beauty whose price is <= p.

Time complexity: Pre-processing O(nlogn) + query: O(qlogn)
Space complexity: O(n)

*/
public class Solution {
    public int[] MaximumBeauty(int[][] items, int[] queries) {
      // sort the items by price, then by beauty descending to make filtering list easier
        Array.Sort(items, (a, b) => a[0]==b[0] ? b[1] - a[1] : a[0] - b[0]);
        
        // LINQ is not quicker
        //var sortedItems = items.OrderBy(x => x[0]).ThenByDescending(x => x[1]).ToArray();
        
        // parallel arrays to hold filtered data
        // only need to keep an increase in beauty and price
        // could keep prices and beauty together, but extra work for binary search
        var prices = new List<int>();
        var beauties = new List<int>();
        prices.Add(items[0][0]);
        beauties.Add(items[0][1]);
        for(int i = 1; i < items.Length; i++){
            if(items[i][1] > beauties[^1]){
                prices.Add(items[i][0]);
                beauties.Add(items[i][1]);
            }
        }

        var answer = new int[queries.Length];
        
        for(int i = 0; i < queries.Length; i++){
            int index = prices.BinarySearch(queries[i]);
            if(index >= 0){
                answer[i] = beauties[index];
            }else{
                index = ~index - 1;
                if(index >= 0) answer[i] = beauties[index];
            }
        }
        
        return answer;
  }
}

public class Solution {
    public int[] MaximumBeauty(int[][] items, int[] queries) {
        //sort items by price
            items = items.OrderBy(x => x[0]).ToArray();
            int[][] arr = new int[items.Length][];//store {price, maxBeauty}
            int beauty = int.MinValue; ;
            for(int i = 0; i < items.Length; i++)
            {
                beauty = Math.Max(beauty, items[i][1]);//update max beauty
                arr[i] = new int[] { items[i][0], beauty };//store it
            }

            var res = new int[queries.Length];//result array
            for (int i=0;i< queries.Length; i++)
            {
                if (queries[i] < arr[0][0])
                {
                    res[i] = 0;//less than min price, assign 0 then continue
                    continue;
                }
                if (queries[i] >= arr[items.Length - 1][0])
                {
                    res[i] = arr[items.Length - 1][1];// >= max price, assign max beauty, then continue
                    continue;
                }

                int left = 0;
                int right = items.Length - 1;
                while (left < right)
                {
				//using binary search, because left always available, set mid the right of center, using (left+right+1)/2
                    int mid = (left + right+1) / 2;
                    if (arr[mid][0] > queries[i])
                    {
                        right = mid - 1;
                    }
                    else
                    {
                        left = mid;
                    }
                }
                res[i] = arr[left][1];
            }
            return res;
  }
}

public class Solution {
    public int[] MaximumBeauty(int[][] itms, int[] q) { 
        int[] r = new int[q.Length];
        
        Array.Sort(itms, new Comparison<int[]>((x,y) => { return x[0] < y[0] ? -1 : (x[0] > y[0] ? 1 : 0); }));
        
        for (int i = 1; i < itms.Length; ++i)
            itms[i][1] = Math.Max(itms[i][1], itms[i - 1][1]);
        
        for (int i = itms.Length - 1; i >= 1; --i) 
            if (itms[i][0] == itms[i - 1][0]) itms[i - 1][1] = itms[i][1];
        
        for (int i = 0; i < q.Length; ++i) 
            r[i] = BinarySearch(itms, q[i]);
        
        return r;
    }
    
    public int BinarySearch(int[][] itms, int p) { 
        int l = 0, r = itms.Length - 1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (itms[m][0] == p) {
                return itms[m][1];
            } else if (p > itms[m][0]) {
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
        
        if (l < 0 || r < 0) return 0;
        else if (l >= itms.Length) return itms[l - 1][1];
        else return itms[l][0] > p ? itms[l - 1][1] : itms[l][1];
  }
}

// 2071. Maximum Number of Tasks You Can Assign
/*Solution: Greedy + Binary Search in Binary Search.
Find the smallest k, s.t. we are NOT able to assign. Then answer is k- 1.

The key is to verify whether we can assign k tasks or not.

Greedy: We want k smallest tasks and k strongest workers.

Start with the hardest tasks among (smallest) k:
1. assign task[i] to the weakest worker without a pill 
(if he can handle the hardest work so far, 
then the stronger workers can handle any simpler tasks left)
2. If 1) is not possible, we find a weakest worker + pill that can handle task[i] 
(otherwise we are wasting workers)
3. If 2) is not possible, impossible to finish k tasks.

Let k = min(n, m)
Time complexity: O((logk)2 * k)
Space complexity: O(k)*/
public class Solution {
    public int MaxTaskAssign(int[] t, int[] w, int p, int s) {
        Array.Sort(t);
        Array.Sort(w);
        
        int l = 0, r = Math.Min(t.Length, w.Length);
        while (l + 1 < r) {
            int m = l + (r - l) / 2;
            if (CanAssign(t, w, p, s, m)) {
                l = m;
            } else {
                r = m;
            }
        }
        
        if (CanAssign(t, w, p, s, r)) return r;
        else return l;
    }
    
    public bool CanAssign(int[] t, int[] w, int p, int s, int cnt) {
        List<int> dq = new List<int>();
        int end = w.Length - 1;
        for (int i = cnt - 1; i >= 0; --i) {
            while (end >= w.Length - cnt && w[end] + s >= t[i]) {
                dq.Add(w[end]);
                end--;
            }
            
            if (dq.Count == 0) return false;
        
            if (dq[0] >= t[i]) {
                dq.RemoveAt(0);
            } else {
                dq.RemoveAt(dq.Count - 1);
                p--;
                if (p < 0) return false;
            }
        }
        
        return true;
    }
}

// 2073. Time Needed to Buy Tickets
/*Solution 1: Simulation
Time complexity: O(n * tickets[k])
Space complexity: O(n) / O(1)*/
public class Solution {
    public int TimeRequiredToBuy(int[] tickets, int k) {
        int n = tickets.Length;
    Queue<KeyValuePair<int, int>> q = new Queue<KeyValuePair<int, int>>();
    for (int i = 0; i < n; ++i)
      q.Enqueue(new KeyValuePair<int, int>(i, tickets[i]));
    int ans = 0;
    while (q.Count != 0) {
      ++ans;
      var (i, t) = q.Peek(); q.Dequeue();      
      if (--t == 0 && k == i) return ans;
      if (t > 0) q.Enqueue(new KeyValuePair<int, int>(i, t));
    }
    return -1;
    }
}
/*Solution 2: Math
Each person before k will have tickets[k] rounds, 
each person after k will have tickets[k] – 1 rounds.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int TimeRequiredToBuy(int[] tickets, int k) {
        int ans = 0;
    for (int i = 0; i < tickets.Length; ++i)
      ans += Math.Min(tickets[i], tickets[k] - Convert.ToInt32(i > k));
    return ans;
    }
}
// 2074. Reverse Nodes in Even Length Groups
/*Solution: List
Reuse ReverseList from 花花酱 LeetCode 206. Reverse Linked List

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
    public ListNode ReverseEvenLengthGroups(ListNode head) {
        ListNode dummy = new ListNode(0, head);
    ListNode prev = dummy;
    ListNode reverse(ListNode head) {     
      ListNode prev = null;
      while (head != null) {
        ListNode next = head.next;
        head.next = prev;
        prev = head;
        head = next;
      }
      return prev;
    };
    
    for (int k = 1; head != null; ++k) {
      ListNode tail = head;
      int l = 1;
      while (l < k && tail != null && tail.next != null ) {
        tail = tail.next;
        ++l;
      }
      ListNode next = tail.next;
      if (l % 2 == 0) {
        tail.next = null;
        prev.next = reverse(head);
        head.next = next;
        prev = head;        
        head = head.next;
      } else {
        prev = tail;
        head = next;
      }
    }
    return dummy.next;
    }
}
// 2075. Decode the Slanted Ciphertext
/*Solution: Simulation
Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public string DecodeCiphertext(string encodedText, int rows) {
        int cols = encodedText.Length / rows;    
    StringBuilder ans = new StringBuilder(encodedText.Length);
    //ans.reserve(encodedText.Length);
    for (int i = 0; i < cols; ++i)
      for (int j = 0; j < rows && i + j < cols; ++j)      
        ans.Append(encodedText[j * cols + j + i]);
    while (ans.Length > 0 && !char.IsLetter(ans[ans.Length - 1])) ans.Length -=1;
    return ans.ToString();
    }
}

// 2076. Process Restricted Friend Requests
/*Solution: Union Find / Brute Force

For each request, check all restrictions.

Time complexity: O(req * res)
Space complexity: O(n)
*/
public class Solution {
    public bool[] FriendRequests(int n, int[][] restrictions, int[][] requests) {
        int[] parents = Enumerable.Range(0, n).ToArray();
    int find(int x) {
      if (parents[x] == x) return x;
      return parents[x] = find(parents[x]);
    };
    bool check(int u, int v) {
      foreach (int[] r in restrictions) {
        int pu = find(r[0]);
        int pv = find(r[1]);
        if ((pu == u && pv == v) || (pu == v && pv == u))
          return false;
      }
      return true;
    };
    List<bool> ans = new List<bool>();
    foreach (int[] r in requests) {      
      int pu = find(r[0]);
      int pv = find(r[1]);
      if (pu == pv || check(pu, pv)) {
        parents[pu] = pv;
        ans.Add(true);
      } else {
        ans.Add(false);
      }
    }
    return ans.ToArray();
    }
}


// 2064. Minimized Maximum of Products Distributed to Any Store
/*Solution: Binary Search
Find the smallest max product s.t. all products can be distribute to <= n stores.

Time complexity: O(nlog(max(q)))
Space complexity: O(1)*/
public class Solution {
    public int MinimizedMaximum(int n, int[] quantities) {
        int l = 1;
    int r = quantities.Max() + 1; //*max_element(begin(quantities), end(quantities)) + 1;
    while (l < r) {
      int cur = 0;
      int m = l + (r - l) / 2;
      foreach (int q in quantities)
        cur += (q + (m - 1)) / m;
      if (cur <= n)
        r = m;
      else
        l = m + 1;
    }
    return l;
    }
}

// 2063. Vowels of All Substrings
/*Solution: Math
For a vowel at index i,
we can choose 0, 1, … i as starting point
choose i, i+1, …, n -1 as end point.
There will be (i – 0 + 1) * (n – 1 – i + 1) possible substrings 
that contains word[i].

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public long CountVowels(string word) {
          long n = word.Length;
    long ans = 0;
    for (long i = 0; i < n; ++i) {
      switch (word[(int)i]) {
        case 'a':
        case 'e':
        case 'i':
        case 'o':
        case 'u':
          ans += (i + 1) * (n - 1 - i + 1);
          break;
      }
    }
    return ans;
    }
}

// 2062. Count Vowel Substrings of a String
/*Solution 1: Brute Force

Time complexity: O(n2)
Space complexity: O(1)*/
public class Solution {
    public int CountVowelSubstrings(string word) {
     int n = word.Length;
    bool check(string s) {
      HashSet<int> seen = new HashSet<int>();
      string vowels = "aeiou";
      foreach (char c in s) {
        if (!vowels.Contains(c)) return false;
        seen.Add(c);
      }
      return seen.Count == 5;
    };
    int ans = 0;
    for (int i = 0; i < n; ++i)
      for (int l = 5; i + l <= n; ++l)
        if (check(word.Substring(i, l))) ++ans;
    return ans;
    }
}

/*Solution 2: Sliding Window / Three Pointers
Maintain a window [i, j] that contain all 5 vowels, find k s.t. [k + 1, i] no longer container 5 vowels.
# of valid substrings end with j will be (k – i).

##aeiouaeioo##
..i....k...j..
i = 3, k = 8, j = 12

Valid substrings are:
aeiouaeioo
.eiouaeioo
..iouaeioo
...ouaeioo
....uaeioo
8 – 3 = 5

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int CountVowelSubstrings(string word) {
      string vowels = "aeiou";
    int ans = 0;
    Dictionary<char, int> m = new Dictionary<char, int>();
    foreach (char c in vowels) m[c] = m.GetValueOrDefault(c, 0);
    for (int i = 0, j = 0, k = 0, v = 0; j < word.Length; ++j) {
      if (m.ContainsKey(word[j]) ) {
        v += Convert.ToInt32(++m[word[j]] == 1);
        while (v == 5)
          v -= Convert.ToInt32(--m[word[k++]] == 0);
        ans += k - i;
      } else {
        foreach (char c in vowels) m[c] = 0;
        v = 0;
        i = k = j + 1;
      }
    }
    return ans;
    }
}
// 2059. Minimum Operations to Convert Number
/*Solution: BFS
Time complexity: O(n*m)
Space complexity: O(m)*/
public class Solution {
    public int MinimumOperations(int[] nums, int start, int goal) {
         int[] seen = new int[1001];
    Queue<int> q = new Queue<int>(); //{{start}};
      q.Enqueue(start);
    for (int ans = 1, s = 1; q.Count != 0; ++ans, s = q.Count) {
      while (s-- > 0) {
        int x = q.Peek(); q.Dequeue();
        foreach (int n in nums)
          foreach (int t in new int[3]{x + n, x - n, x ^ n})
            if (t == goal) 
              return ans;
            else if (t < 0 || t > 1000 || seen[t]++ > 0)
              continue;
            else
              q.Enqueue(t);
      }
    }
    return -1;
    }
}

// 2058. Find the Minimum and Maximum Number of Nodes Between Critical Points
/*Solution: One Pass
Track the first and last critical points.

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
    public int[] NodesBetweenCriticalPoints(ListNode head) {
        ListNode p = null;
    int min_dist = Int32.MaxValue;
    int max_dist = Int32.MinValue;    
    for (int i = 0, l = -1, r = -1; head != null; ++i, p = head, head = head.next) {
      if (p != null && head.next != null) {
        if ((head.val > p.val && head.val > head.next.val) 
            || (head.val < p.val && head.val < head.next.val)) {
          if (r != -1) {
            min_dist = Math.Min(min_dist, i - r);
            max_dist = Math.Max(max_dist, i - l);
          } else {
            l = i;
          }          
          r = i;          
        }
      }
    }
    return new int[2]{ min_dist == Int32.MaxValue ? -1 : min_dist, 
             max_dist == Int32.MinValue ? -1 : max_dist };
    }
}

// 2057. Smallest Index With Equal Value
/*Solution: Brute Force
Time complexity: O(n)
Space complexity: O(1)
*/
public class Solution {
    public int SmallestEqual(int[] nums) {
        for (int i = 0; i < nums.Length; ++i)
      if (i % 10 == nums[i]) return i;
    return -1;
    }
}

// 2055. Plates Between Candles
/*Solution: Binary Search
Store the indices of all candles into an array idx.

For each query q:
1. Find the left most candle whose index is greater or equal to left as l.
2. Find the left most candle whose index is greater than right, 
choose the previous candle as r.

[idx[l], idx[r]] is the range that are elements between two candles , 
there are (idx[r] – idx[l] + 1) elements in total and there are (r – l + 1) 
candles in the range. 
So the number of plates is (idx[r] – idx[l] + 1) – (r – l + 1) 
or (idx[r] – idx[l]) – (r – l)

Time complexity: O(qlogn)
Space complexity: O(n)*/
public class Solution {
    public int[] PlatesBetweenCandles(string s, int[][] queries) {
         List<int> idx = new List<int>();
    for (int i = 0; i < s.Length; ++i)
      if (s[i] == '|') idx.Add(i);
    List<int> ans = new List<int>();
    foreach (int[] q in queries) {
        int r = idx.BinarySearch(q[1]+1); // directly use C# binary search find upperbound, need to r - 1 later!
        if (r < 0) { r = ~r;} r -= 1;
         int l = idx.BinarySearch(q[0]); // directly use C# binary search find lowerbound
            if (l < 0) { l = ~l;}
      ans.Add(l >= r ? 0 : (idx[r] - idx[l]) - (r - l));
    }
    return ans.ToArray();
  }
}

public class Solution {
    public int[] PlatesBetweenCandles(string s, int[][] queries) {
         List<int> idx = new List<int>();
    for (int i = 0; i < s.Length; ++i)
      if (s[i] == '|') idx.Add(i);
    List<int> ans = new List<int>();
    foreach (int[] q in queries) {
      int l =  lowerBound(idx, q[0]);
      int r = upperBound(idx, q[1])  - 1;      
      ans.Add(l >= r ? 0 : (idx[r] - idx[l]) - (r - l));
    }
    return ans.ToArray();
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

// 2054. Two Best Non-Overlapping Events
/*This problem can be solved with the help of heap.

First sort all events by start time. 
If start time of two events are equal, sort them by end time.
Then take a priority queue that takes an array containing [endtime, value]. 
Priority queue will sort elements on the basis of end time.
Iterate through events, for each event e, 
calculate maximum value from all events that ends before e[0] (i.e. start time). 
Let's store this value in maxVal variable.
Now answer will be ans = max(ans, e[2] + maxVal).*/
public class Solution {
    public int MaxTwoEvents(int[][] events) {
        int n = events.Length;
        Array.Sort(events, (a, b) => a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
         PriorityQueue<int[],int[]> queue = new PriorityQueue<int[],int[]>(Comparer<int[]>.Create((a, b) => a[0] - b[0]));
        
        int maxVal = 0, ans = 0;
        foreach(int[] e in events){            
            int start = e[0];
            while(queue.Count != 0){
                if(queue.Peek()[0] >= start)
                    break;
                int[] eve = queue.Dequeue();
                maxVal = Math.Max(maxVal, eve[1]);
            }
            ans = Math.Max(ans, e[2] + maxVal);
            queue.Enqueue(new int[]{e[1], e[2]},new int[]{e[1], e[2]});
        }
        
        return ans;
    }
}
/*Solution: Sort + Heap
Sort events by start time, process them from left to right.

Use a min heap to store the events processed so far, 
a variable cur to track the max value of a non-overlapping event.

For a given event, pop all non-overlapping events 
whose end time is smaller than its start time and update cur.

ans = max(val + cur)

Time complexity: O(nlogn)
Space complexity: O(n)*/
public class Solution {
    public int MaxTwoEvents(int[][] events) {
    Array.Sort(events, (a, b) => a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
    PriorityQueue<KeyValuePair<int,int>, KeyValuePair<int,int>> q = new PriorityQueue<KeyValuePair<int,int>, KeyValuePair<int,int>>(Comparer<KeyValuePair<int,int>>.Create((a, b) => a.Key - b.Key));
    int ans = 0;
    int cur = 0;
    foreach (int[] e in events) {
      while (q.Count != 0 && q.Peek().Key < e[0]) {
        cur = Math.Max(cur, q.Peek().Value);
        q.Dequeue();
      }
      ans = Math.Max(ans, cur + e[2]);
      q.Enqueue(new KeyValuePair<int,int>(e[1], e[2]),new KeyValuePair<int,int>(e[1], e[2]));
    }
    return ans;
    }
}
// 2053. Kth Distinct String in an Array
/*Solution: Hashtable
Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public string KthDistinct(string[] arr, int k) {
        Dictionary<string, int> m = new Dictionary<string, int> ();
    foreach (string s in arr)
      m[s] = m.GetValueOrDefault(s,0) + 1;    
    foreach (string s in arr)
      if (m[s] == 1 && --k == 0) return s;    
    return "";
    }
}

// 2049. Count Nodes With the Highest Score
/*Solution: Recursion
Write a function that returns the element of a subtree rooted at node.

We can compute the score based on:
1. size of the subtree(s)
2. # of children

Root is a special case whose score is max(c[0], 1) * max(c[1], 1).

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int CountHighestScoreNodes(int[] parents) {
        int n = parents.Length;
    long high = 0;
    int ans = 0;
    List<int>[] tree = new List<int>[n];
    for(int i=0; i<n; i++)
        tree[i] = new List<int>();
    for (int i = 1; i < n; ++i){
       tree[parents[i]].Add(i);
    }
      
    int dfs(int node) {      
      long[] c = new long[2]{0, 0};
      for (int i = 0; i < tree[node].Count; ++i)
        c[i] = dfs(tree[node][i]);
      long score = 0;
      if (node == 0) // case #1: root
        score = Math.Max(c[0], 1L) * Math.Max(c[1], 1L);
      else if (tree[node].Count == 0) // case #2: leaf
        score = n - 1;
      else if (tree[node].Count == 1) // case #3: one child
        score = c[0] * (n - c[0] - 1);
      else // case #4: two children
        score = c[0] * c[1] * (n - c[0] - c[1] - 1);
      if (score > high) {
        high = score;
        ans = 1;
      } else if (score == high) {
        ++ans;
      }
      return (int)(1 + c[0] + c[1]);
    };
    dfs(0);
    return ans;
    }
}

// 2022. Convert 1D Array Into 2D Array
/*Solution: Brute Force
the i-th element in original array will have index (i//n, i % n) in the 2D array.

Time complexity: O(n*m)
Space complexity: O(n*m)*/
public class Solution {
    public int[][] Construct2DArray(int[] original, int m, int n) {
        if (original.Length != m * n) return new int[0][];
    int[][] ans = new int[m][];
         for (int i = 0; i < m; ++i)
              ans[i] = new int[n];
    for (int i = 0; i < m * n; ++i)
      ans[i / n][i % n] = original[i];
    return ans;
    }
}

// 2027. Minimum Moves to Convert String
/*Solution: Straight Forward
if s[i] == ‘X’, change s[i], s[i + 1] and s[i + 2] to ‘O’.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MinimumMoves(string s) {
         int n = s.Length;
    int ans = 0;
    for (int i = 0; i < n; ++i) {
      if (s[i] == 'X') {
        ans += 1;
        i += 2;
      }
    }
    return ans;
    }
}

// 2028. Find Missing Observations
/*Solution: Math & Greedy
Total sum = (m + n) * mean
Left = Total sum – sum(rolls) = (m + n) * mean – sum(rolls)
If left > 6 * n or left < 1 * n, then there is no solution.
Otherwise, we need to distribute Left into n rolls.
There are very ways to do that, one of them is even distribution, e.g. using the average number as much as possible, and use avg + 1 to fill the gap.
Compute the average and reminder: x = left / n, r = left % n.
there will be n – r of x and r of x + 1 in the output array.

e.g. [1, 5, 6], mean = 3, n = 4
Total sum = (3 + 4) * 3 = 21
Left = 21 – (1 + 5 + 6) = 9
x = 9 / 4 = 2, r = 9 % 4 = 1
Ans = [2, 2, 2, 2+1] = [2,2,2,3]

Time complexity: O(m + n)
Space complexity: O(1)*/
public class Solution {
    public int[] MissingRolls(int[] rolls, int mean, int n) {
        int m = rolls.Length;    
    int t =  0; Array.ForEach(rolls, i => t += i);
    int left = mean * (m + n) - t;
    if (left > 6 * n || left < n) return new int[0];    
    int[] ans = new int[n]; Array.Fill(ans, left / n);
    for (int i = 0; i < left % n; ++i) ++ans[i];
    return ans;
    }
}

// 2032. Two Out of Three
/*Solution: Hashmap / Bitmask
s[x] := bitmask of x in all array[i]

s[x] = 101 => x in array0 and array2

Time complexity: O(n1 + n2 + n3)
Space complexity: O(n1 + n2 + n3)*/
using System.Numerics;
public class Solution {
    public IList<int> TwoOutOfThree(int[] nums1, int[] nums2, int[] nums3) {
        Dictionary<int, int> s = new Dictionary<int, int>();    
    foreach (int x in nums1) s[x] = s.GetValueOrDefault(x , 0) | 1;
    foreach (int x in nums2) s[x] = s.GetValueOrDefault(x , 0) | 2;
    foreach (int x in nums3) s[x] = s.GetValueOrDefault(x , 0) | 4;
     IList<int> ans = new List<int>();
    foreach (var (x, v) in s)
      if (BitOperations.PopCount((uint)v) >= 2) ans.Add(x);
    return ans;
    }
}

// 2033. Minimum Operations to Make a Uni-Value Grid
/*Solution: Median
To achieve minimum operations, the uni-value must be the median of the array.

Time complexity: O(m*n)
Space complexity: O(m*n)*/
public class Solution {
    public int MinOperations(int[][] grid, int x) {
        int mn = grid.Length * grid[0].Length;    
    List<int> nums = new List<int>();
      int index = 0;
    foreach (int[] row in grid)
        nums.AddRange(row.ToList());
    //Array.Sort(nums);
        nums.Sort();
    int median = nums[mn / 2];
    int ans = 0;
    foreach (int v in nums) {       
      if ((Math.Abs(v - median) % x) != 0) return -1;
      ans += Math.Abs(v - median) / x;
    }
    return ans;
    }
}

public class Solution {
    public int MinOperations(int[][] grid, int x) {
        int mn = grid.Length * grid[0].Length;    
    int[] nums = new int[mn];
    //nums.reserve(mn);
      int index = 0;
    for (int i = 0; i < grid.Length; i++) {
            for (int j = 0; j < grid[0].Length; j++) {
                nums[index++] = grid[i][j];
            }
        }
    //nth_element(begin(nums), begin(nums) + mn / 2, end(nums));
    Array.Sort(nums);
    int median = nums[mn / 2];
    int ans = 0;
    foreach (int v in nums) {       
      if ((Math.Abs(v - median) % x) != 0) return -1;
      ans += Math.Abs(v - median) / x;
    }
    return ans;
    }
}

// 2037. Minimum Number of Moves to Seat Everyone
/*Solution: Greedy
Sort both arrays, move students[i] to seats[i].

Time complexity: O(nlogn)
Space complexity: O(1)*/
public class Solution {
    public int MinMovesToSeat(int[] seats, int[] students) {
        Array.Sort(seats);
    Array.Sort(students);
    int ans = 0;
    for (int i = 0; i < seats.Length; ++i)
      ans += Math.Abs(seats[i] - students[i]);
    return ans;
    }
}

// 2038. Remove Colored Pieces if Both Neighbors are the Same Color
/*Solution: Counting
Count how many ‘AAA’s and ‘BBB’s.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public bool WinnerOfGame(string colors) {
         int n = colors.Length;
    int count = 0;
    for (int i = 1; i < n - 1; ++i)
      if (colors[i - 1] == colors[i] &&
          colors[i] == colors[i + 1])
        count += (colors[i] == 'A' ? 1 : -1);    
    return count > 0;
    }
}

// 2039. The Time When the Network Becomes Idle
/*Solution: Shortest Path
Compute the shortest path from node 0 to rest of the nodes using BFS.

Idle time for node i = (dist[i] * 2 – 1) / patince[i] * patience[i] + dist[i] * 2 + 1

Time complexity: O(E + V)
Space complexity: O(E + V)*/
public class Solution {
    public int NetworkBecomesIdle(int[][] edges, int[] patience) {
         int n = patience.Length;
    List<int>[] g = new List<int>[n];
    foreach (int[] e in edges) {
        if (g[e[0]] == null) {g[e[0]] = new List<int>();}
         if (g[e[1]] == null) {g[e[1]] = new List<int>();}
      g[e[0]].Add(e[1]);
      g[e[1]].Add(e[0]);
    }
    int[] ts = new int[n]; Array.Fill(ts, -1);
    ts[0] = 0;
    Queue<int> q = new Queue<int>();
    q.Enqueue(0);
    while (q.Count != 0) {      
      int u = q.Peek(); q.Dequeue();        
      foreach (int v in g[u]) {
        if (ts[v] != -1) continue;
        ts[v] = ts[u] + 1;
        q.Enqueue(v);
      }
    }
    int ans = 0;
    for (int i = 1; i < n; ++i) {      
      int t = (ts[i] * 2 - 1) / patience[i] * patience[i] + ts[i] * 2 + 1;
      ans = Math.Max(ans, t);
    }    
    return ans;
    }
}

// 2042. Check if Numbers Are Ascending in a Sentence
/*Solution: String

Time complexity: O(n)
Space complexity: O(1)
*/
public class Solution {
    public bool AreNumbersAscending(string s) {
        String[] ss = s.Split(" ").ToArray();
    int last = -1;
    foreach (string token in ss) {
      if (Char.IsDigit(token[0])) {
        int num = Convert.ToInt32(token);
        if (num <= last) return false;
        last = num;
      }
    }
    return true;
    }
}

// 2043. Simple Bank System
/*Solution: Simulation
Time complexity: O(1) per operation
Space complexity: O(n) for n accounts*/
public class Bank {

    private int n_;
    private long[] balance_;
    
    public Bank(long[] balance) {
         n_ = balance.Length; balance_ = balance;
    }
    
    public bool Transfer(int account1, int account2, long money) {
    if (account1 <= 0 || account1 > n_) return false;
    if (account2 <= 0 || account2 > n_) return false;
    if (balance_[account1 - 1] < money) return false;
    balance_[account1 - 1] -= money;
    balance_[account2 - 1] += money;
    return true;
    }
    
    public bool Deposit(int account, long money) {
         if (account <= 0 || account > n_) return false;
        balance_[account - 1] += money;
        return true;
    }
    
    public bool Withdraw(int account, long money) {
        if (account <= 0 || account > n_ || balance_[account - 1] < money) 
            return false;
        balance_[account - 1] -= money;
        return true;
    }
}

 
/**
 * Your Bank object will be instantiated and called as such:
 * Bank obj = new Bank(balance);
 * bool param_1 = obj.Transfer(account1,account2,money);
 * bool param_2 = obj.Deposit(account,money);
 * bool param_3 = obj.Withdraw(account,money);
 */

// 2044. Count Number of Maximum Bitwise-OR Subsets
/*Solution: Brute Force
Try all possible subsets

Time complexity: O(n*2n)
Space complexity: O(1)*/
public class Solution {
    public int CountMaxOrSubsets(int[] nums) {
        int n = nums.Length;
    int max_or = 0;
    int count = 0;
    for (int s = 0; s < 1 << n; ++s) {
      int cur_or = 0;
      for (int i = 0; i < n; ++i)
        if ((s >> i & 1) > 0) cur_or |= nums[i];
      if (cur_or > max_or) {
        max_or = cur_or;
        count = 1;
      } else if (cur_or == max_or) {
        ++count;
      }
    }
    return count;
    }
}

// 2045. Second Minimum Time to Reach Destination
/*Solution: Best first search
Since we’re only looking for second best, to avoid TLE, 
for each vertex, keep two best time to arrival is sufficient.

Time complexity: O(2ElogE)
Space complexity: O(V+E)*/
public class Solution {
    public int SecondMinimum(int n, int[][] edges, int time, int change) {
        List<int>[] g = new List<int>[n + 1];
    foreach (int[] e in edges) {
        if (g[e[0]] == null) {g[e[0]] = new List<int>();}
         if (g[e[1]] == null) {g[e[1]] = new List<int>();}
        
      g[e[0]].Add(e[1]);
      g[e[1]].Add(e[0]);
    }
    PriorityQueue<KeyValuePair<int, int>, KeyValuePair<int, int>> q = new PriorityQueue<KeyValuePair<int, int>, KeyValuePair<int, int>>(Comparer<KeyValuePair<int,int>>.Create((a, b) => b.Key - a.Key)); // {-t, node}
    q.Enqueue( new KeyValuePair<int, int>(0, 1), new KeyValuePair<int, int>(0, 1));
    int min_time = -1;
    List<int>[] ts = new List<int>[n + 1];
    for (int i = 0; i < n+1; i++) {
         if (ts[i] == null) {ts[i] = new List<int>();}
    }
    ts[1].Add(0);
    while (true) {
      int t = -q.Peek().Key;
      int u = q.Peek().Value;
      if (u == n) {
        if (min_time == -1 || t == min_time) { 
          min_time = t;
        } else {
          return t;
        }
      }
      q.Dequeue();
      foreach (int v in g[u]) {
       
        int tt = ((((t / change) & 1) > 0 ) ? (t + change - t % change) : t) + time;
        if ((ts[v].Count > 0 && tt == ts[v][ts[v].Count - 1]) || ts[v].Count >= 2) continue;
         
        ts[v].Add(tt);        
        q.Enqueue( new KeyValuePair<int, int>(-tt, v),new KeyValuePair<int, int>(-tt, v));        
      }
    }
    return -1;
    }
}

// 2047. Number of Valid Words in a Sentence
/*Solution 1: Brute Force
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int CountValidWords(string sentence) {
    int ans = 0;
        string[] words = sentence.Split(' ');      
    foreach(string word in words){
    //while (ss.Length > word.Length) { 
        if(word.Length == 0) continue;
        if(!char.IsLetter(word[0]) && word.Length != 1) continue;

      bool valid = true;      
      int hyphen = 0;
      int punctuation = 0;
      char p = ' ';
      foreach (char c in word) {
        if (c == '-') {          
          if (++hyphen > 1 || !char.IsLetter(p)) {
            valid = false;
            break;
          }
        } else if (c == '!' || c == '.' || c == ',') {
          if (++punctuation > 1 || p == '-') {
            valid = false;
            break;
          }
        } else if (char.IsLetter(c)) {          
          if (punctuation > 0) {
            valid = false;
            break;
          }
        } else {
          valid = false;
          break;
        }
        p = c;
      }
      if (word[word.Length - 1] == '-') 
        valid = false;
      if (valid) ++ans;      
    }
    return ans;
    }
}

//Regex
using System.Text.RegularExpressions;
public class Solution {
    public int CountValidWords(string sentence)  => sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries).Count(word => Regex.IsMatch(word, "^[a-z]*([a-z][-][a-z]+)?([.,!])?$"));
}

//Regex
using System.Text.RegularExpressions;
public class Solution {
    public int CountValidWords(string sentence) {
        var regex = new Regex(@"^([a-z]+(-?[a-z]+)?)?(!|\.|,)?$");
        var words = sentence.Split(' ');
        var result = 0;
        
        foreach(var word in words)
            if(!string.IsNullOrEmpty(word) && regex.Match(word).Success)
                result++;
        
        return result;
    }
}

// 2048. Next Greater Numerically Balanced Number
/*Solution: Permutation
Time complexity: O(log(n)!)
Space complexity: O(log(n)) ?*/
public class Solution {
    public int NextBeautifulNumber(int n) {
        string t = n.ToString();
    List<int> nums = new List<int>(){
                      1, 22, 122, 212, 221, 333, 1333, 3133, 3313, 3331, 4444, 14444, 22333, 23233, 23323, 23332, 32233, 32323, 32332, 33223, 33232, 33322,
		41444, 44144, 44414, 44441, 55555, 122333, 123233, 123323, 123332, 132233, 132323, 132332, 133223, 133232, 133322, 155555,
		212333, 213233, 213323, 213332, 221333, 223133, 223313, 223331, 224444, 231233, 231323, 231332, 232133, 232313, 232331, 233123,
		233132, 233213, 233231, 233312, 233321, 242444, 244244, 244424, 244442, 312233, 312323, 312332, 313223, 313232, 313322, 321233,
		321323, 321332, 322133, 322313, 322331, 323123, 323132, 323213, 323231, 323312, 323321, 331223, 331232, 331322, 332123, 332132,
		332213, 332231, 332312, 332321, 333122, 333212, 333221, 422444, 424244, 424424, 424442, 442244, 442424, 442442, 444224, 444242,
		444422, 515555, 551555, 555155, 555515, 555551, 666666, 1224444};
    int ans = 1224444;
    foreach (int num in nums) {
      string s = num.ToString();
      if (s.Length < t.Length) continue;
      else if (s.Length > t.Length) {
        ans = Math.Min(ans, num);
      } else { // same length 
        
          if (Convert.ToInt32(s) > Convert.ToInt32(t)) ans = Math.Min(ans, Convert.ToInt32(s));
        ;
      }
    }
    return ans;
    }
}

public class Solution {
     static int[] preCalcBalanced = new int[]{
        1, 22, 122, 212, 221, 333, 1333, 3133, 3313, 3331, 4444, 14444, 22333, 23233, 23323, 23332, 32233, 32323, 32332, 33223, 33232, 33322,
		41444, 44144, 44414, 44441, 55555, 122333, 123233, 123323, 123332, 132233, 132323, 132332, 133223, 133232, 133322, 155555,
		212333, 213233, 213323, 213332, 221333, 223133, 223313, 223331, 224444, 231233, 231323, 231332, 232133, 232313, 232331, 233123,
		233132, 233213, 233231, 233312, 233321, 242444, 244244, 244424, 244442, 312233, 312323, 312332, 313223, 313232, 313322, 321233,
		321323, 321332, 322133, 322313, 322331, 323123, 323132, 323213, 323231, 323312, 323321, 331223, 331232, 331322, 332123, 332132,
		332213, 332231, 332312, 332321, 333122, 333212, 333221, 422444, 424244, 424424, 424442, 442244, 442424, 442442, 444224, 444242,
		444422, 515555, 551555, 555155, 555515, 555551, 666666, 1224444 };
    
    public int NextBeautifulNumber(int n) {
        
        int index = Array.BinarySearch(preCalcBalanced, n+1);
        if( index < 0) index = ~index;
        return preCalcBalanced[index];
    }
}

public class Solution {
     static int[] preCalcBalanced = new int[]{
        1, 22, 122, 212, 221, 333, 1333, 3133, 3313, 3331, 4444, 14444, 22333, 23233, 23323, 23332, 32233, 32323, 32332, 33223, 33232, 33322,
		41444, 44144, 44414, 44441, 55555, 122333, 123233, 123323, 123332, 132233, 132323, 132332, 133223, 133232, 133322, 155555,
		212333, 213233, 213323, 213332, 221333, 223133, 223313, 223331, 224444, 231233, 231323, 231332, 232133, 232313, 232331, 233123,
		233132, 233213, 233231, 233312, 233321, 242444, 244244, 244424, 244442, 312233, 312323, 312332, 313223, 313232, 313322, 321233,
		321323, 321332, 322133, 322313, 322331, 323123, 323132, 323213, 323231, 323312, 323321, 331223, 331232, 331322, 332123, 332132,
		332213, 332231, 332312, 332321, 333122, 333212, 333221, 422444, 424244, 424424, 424442, 442244, 442424, 442442, 444224, 444242,
		444422, 515555, 551555, 555155, 555515, 555551, 666666, 1224444 };
    
    public int NextBeautifulNumber(int n) {
        
        var list = preCalcBalanced;
	var index = GetIndex(n, list);
	var rs = list[index];
	return rs;
}
private int GetIndex(int n, int[] list)
{
	var index0 = 0;
	if (n < list[index0]) return index0;
	var index1 = list.Length - 1;
	while(index1 - index0 > 1)
	{
		var indexMid = (index0 + index1) / 2;
		if (n < list[indexMid])
		{
			index1 = indexMid;
		}
		else
		{
			index0 = indexMid;
		}
	}
	return index1;
}
}
// 1906. Minimum Absolute Difference Queries
/*Solution: Binary Search
Since the value range of num is quiet small [1~100], 
we can store the indices for each value.
[2, 1, 2, 2, 3] => {1: [1], 2: [0, 2, 3]: 3: [4]}.

For each query, we try all possible value b. 
Check whether b is the query range using binary search, 
we also keep tracking the previous available value a, ans will be min{b – a}.

Time complexity: O(n + q * 100 * log(n))
Space complexity: O(n)*/
public class Solution {
    public int[] MinDifference(int[] nums, int[][] queries) {
        int kMax = 100;
    int n = nums.Length;
    List<int>[] idx = new List<int>[kMax + 1];
    for (int i = 0; i < n; ++i){
         if (idx[nums[i]] == null) {idx[nums[i]] = new List<int>();}
        idx[nums[i]].Add(i);
    }
      
    List<int> ans =  new List<int>() ;
    foreach (int[] q in queries) {      
      int diff = Int32.MaxValue;
      for (int a = 0, b = 1; b <= kMax; ++b) {
        int it = lowerBound(idx[b], q[0]);
        if (it == -1  || it == idx[b].Count || idx[b][it] > q[1]) continue;
        if (a != 0) diff = Math.Min(diff, b - a);
        a = b;
      }
      ans.Add(diff == Int32.MaxValue ? -1 : diff);
    }
    return ans.ToArray();
    }
    
    public int lowerBound(List<int> nums, int target ) {
         if(nums == null) {
            return -1;
        }
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
}


// 1899. Merge Triplets to Form Target Triplet
/*Solution: Greedy
Exclude those bad ones (whose values are greater than x, y, z), 
check the max value for each dimension or 
whether there is x, y, z for each dimension.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public bool MergeTriplets(int[][] triplets, int[] target) {
        int[] ans = new int[3];
    foreach (int[] c in triplets)
      if (c[0] <= target[0] && c[1] <= target[1] && c[2] <= target[2])
        for (int i = 0; i < 3; ++i)
          ans[i] |= Convert.ToInt32(c[i] == target[i]);
    return  Convert.ToBoolean(ans[0]) && Convert.ToBoolean(ans[1]) && Convert.ToBoolean(ans[2]);
    }
}
// 1900. The Earliest and Latest Rounds Where Players Compete
/*Solution 1: Simulation using recursion
All possible paths,
Time complexity: O(n2*2n)
Space complexity: O(logn)

dfs(s, i, j, d) := let i battle with j at round d, given s (binary mask of dead players).*/
public class Solution {
    public int[] EarliestAndLatest(int n, int firstPlayer, int secondPlayer) {
        int min_r = Int32.MaxValue, max_r = Int32.MinValue;

    void dfs (int mask, int round, int i, int j, int first, int second) {
    if (i >= j)
        dfs(mask, round + 1, 0, 27, first, second);
    else if ((mask & (1 << i)) == 0)
        dfs(mask, round, i + 1, j, first, second);
    else if ((mask & (1 << j)) == 0)
        dfs(mask, round, i, j - 1, first, second);
    else if (i == first && j == second) {
        min_r = Math.Min(min_r, round);
        max_r = Math.Max(max_r, round);
    }
    else {
        if (i != first && i != second)
            dfs(mask ^ (1 << i), round, i + 1, j - 1, first, second);
        if (j != first && j != second)
            dfs(mask ^ (1 << j), round, i + 1, j - 1, first, second);
    }
    };
    dfs((1 << n) - 1, 1, 0, 27, firstPlayer - 1, secondPlayer - 1);
    return new int[2] { min_r, max_r };
  }
}

// 1903. Largest Odd Number in String
/*Solution: Find right most odd digit
We just need to find the right most digit that is odd, answer will be num[0:r].

Answer must start with num[0].
Proof:
Assume the largest number is num[i:r] i > 0, we can always extend to the left, e.g. num[i-1:r] which is also an odd number and it’s larger than num[i:r] which contradicts our assumption. Thus the largest odd number (if exists) must start with num[0].

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public string LargestOddNumber(string num) {
         for (int i = num.Length - 1; i >= 0; --i)
      if (((num[i] - '0') & 1) > 0) return num.Substring(0, i + 1);
    return "";
    }
}

// 1904. The Number of Full Rounds You Have Played
/*Solution: String / Simple math
ans = max(0, floor(end / 15) – ceil(start / 15))

Tips:

Write a reusable function to parse time to minutes.
a / b for floor, (a + b – 1) / b for ceil
Time complexity: O(1)
Space complexity: O(1)*/
public class Solution {
    public int NumberOfRounds(string loginTime, string logoutTime) {
         int parseTime(string t) {
      return ((t[0] - '0') * 10 + (t[1] - '0')) * 60 + (t[3] - '0') * 10 + t[4] - '0';
    };
    int m1 = parseTime(loginTime);
    int m2 = parseTime(logoutTime);
    if (m2 < m1) m2 += 24 * 60;
    return Math.Max(0, m2 / 15 - (m1 + 14) / 15);
    }
}

// 1897. Redistribute Characters to Make All Strings Equal
/*Solution: Hashtable
Count the frequency of each character, 
it must be a multiplier of n such that we can evenly distribute it to all the words.
e.g. n = 3, a = 9, b = 6, c = 3, each word will be “aaabbc”.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public bool MakeEqual(string[] words) {
        int[] freq = new int[26];
    foreach (string word in words)
      foreach (char c in word)
        ++freq[c - 'a'];    
    foreach (int f in freq)
      if ((f % words.Length) > 0 ) return false;        
    return true;
    }
}

// 1896. Minimum Cost to Change the Final Value of Expression
/*Solution: DP, Recursion / Simulation w/ Stack
For each expression, stores the min cost to change value to 0 and 1.

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int MinOperationsToFlip(string expression) {
        Stack<int[]> s = new  Stack<int[]>();
    s.Push(new int[3]{0, 0, 0});
    foreach (char e in expression) {
      if (e == '(')
        s.Push(new int[3]{0, 0, 0});
      else if (e == '&' || e == '|')
        s.Peek()[2] = e;
      else {        
        if (Char.IsDigit(e)) s.Push(new int[3]{Convert.ToInt32(e != '0'), Convert.ToInt32(e != '1'), 0});
         var  arr1 = s.Peek(); s.Pop(); // (r0, r1, _)
         var  arr2 = s.Peek(); s.Pop(); // (l0, l1, op)
          int r0 = arr1[0], r1 = arr1[1], _ = arr1[2];
          int l0 = arr2[0], l1 = arr2[1], op = arr2[2];
        if (op == '&') {
          s.Push(new int[3]{Math.Min(l0, r0),
                  Math.Min(l1 + r1, Math.Min(l1, r1) + 1),
                  0});
        } else if (op == '|') {
          s.Push(new int[3]{Math.Min(l0 + r0, Math.Min(l0, r0) + 1),
                  Math.Min(l1, r1),
                  0});
        } else {
          s.Push(new int[3]{r0, r1, 0});
        }
      }
    }
    return Math.Max(s.Peek()[0], s.Peek()[1]);
    }
}
// tuple
public class Solution {
    public int MinOperationsToFlip(string expression) {
        Stack<(int a0, int a1, int a2)> s = new Stack<(int, int, int)>();
    s.Push((0, 0, 0));
    foreach (char e in expression) {
      if (e == '(')
        s.Push((0, 0, 0));
      else if (e == '&' || e == '|'){
           var (a0, a1, a2) = s.Pop();
       s.Push((a0, a1, e));
      }
         
      else {        
        if (Char.IsDigit(e)) s.Push((Convert.ToInt32(e != '0'), Convert.ToInt32(e != '1'), 0));
         var (r0, r1, _) = s.Peek(); s.Pop(); // (r0, r1, _)
         var (l0, l1, op) = s.Peek(); s.Pop(); // (l0, l1, op)
          //int r0 = arr1[0], r1 = arr1[1], _ = arr1[2];
          //int l0 = arr2[0], l1 = arr2[1], op = arr2[2];
        if (op == '&') {
          s.Push((Math.Min(l0, r0),
                  Math.Min(l1 + r1, Math.Min(l1, r1) + 1),
                  0));
        } else if (op == '|') {
          s.Push((Math.Min(l0 + r0, Math.Min(l0, r0) + 1),
                  Math.Min(l1, r1),
                  0));
        } else {
          s.Push((r0, r1, 0));
        }
      }
    }
    return Math.Max(s.Peek().a0, s.Peek().a1);
    }
}
// 1895. Largest Magic Square
/*Solution: Brute Force w/ Prefix Sum
Compute the prefix sum for each row and each column.

And check all possible squares.

Time complexity: O(m*n*min(m,n)2)
Space complexity: O(m*n)

*/
public class Solution {
    public int MinOperationsToFlip(string expression) {
        Stack<(int a0, int a1, int a2)> s = new Stack<(int, int, int)>();
    s.Push((0, 0, 0));
    foreach (char e in expression) {
      if (e == '(')
        s.Push((0, 0, 0));
      else if (e == '&' || e == '|'){
           var (a0, a1, a2) = s.Pop();
       s.Push((a0, a1, e));
      }
         
      else {        
        if (Char.IsDigit(e)) s.Push((Convert.ToInt32(e != '0'), Convert.ToInt32(e != '1'), 0));
         var (r0, r1, _) = s.Peek(); s.Pop(); // (r0, r1, _)
         var (l0, l1, op) = s.Peek(); s.Pop(); // (l0, l1, op)
          //int r0 = arr1[0], r1 = arr1[1], _ = arr1[2];
          //int l0 = arr2[0], l1 = arr2[1], op = arr2[2];
        if (op == '&') {
          s.Push((Math.Min(l0, r0),
                  Math.Min(l1 + r1, Math.Min(l1, r1) + 1),
                  0));
        } else if (op == '|') {
          s.Push((Math.Min(l0 + r0, Math.Min(l0, r0) + 1),
                  Math.Min(l1, r1),
                  0));
        } else {
          s.Push((r0, r1, 0));
        }
      }
    }
    return Math.Max(s.Peek().a0, s.Peek().a1);
    }
}

// 1894. Find the Student that Will Replace the Chalk
/*Solution: Math
Sum up all the students. k %= sum to skip all the middle rounds.

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int ChalkReplacer(int[] chalk, int k) {
         long sum = 0;  Array.ForEach(chalk, i => sum += i);
     
        long l = k % sum;
        
        for(int i=0; i<chalk.Length;i++)
        {
            if(l >= chalk[i])
                l = l - chalk[i];
            else
                return i;
        }
      //if ((k -= chalk[i]) < 0) return i;
    return -1;
    }
}

public class Solution {
    public int ChalkReplacer(int[] chalk, int k) {
         long sum = 0;  Array.ForEach(chalk, i => sum += i);
     
        long l = k % sum;
        
        for(int i=0; i<chalk.Length;i++)
        {
           if ((l -= chalk[i]) < 0) return i;
        }
      //if ((k -= chalk[i]) < 0) return i;
    return -1;
    }
}

// 1893. Check if All the Integers in a Range Are Covered
/*Solution 1: Hashtable
Time complexity: O(n * (right – left))
Space complexity: O(right – left)

*/
public class Solution {
    public bool IsCovered(int[][] ranges, int left, int right) {
    HashSet<int> s = new HashSet<int>();
    foreach (int[] range in ranges)
      for (int i = Math.Max(left, range[0]); i <= Math.Min(right, range[1]); ++i)
        s.Add(i);    
    return s.Count == right - left + 1;     
    }
}

public class Solution {
    public bool IsCovered(int[][] ranges, int left, int right) {

       Array.Sort(ranges, (x,y) =>x[0]-y[0]);
	foreach(int[] range in ranges) 
		if(left >= range[0] && left <= range[1])
			left = range[1] + 1;
	return left > right;
    }
}
// 1887. Reduction Operations to Make the Array Elements Equal
/*Solution: Math
Input: [5,4,3,2,1]
[5,4,3,2,1] -> [4,4,3,2,1] 5->4, 1 op
[4,4,3,2,1] -> [3,3,3,2,1] 4->3, 2 ops
[3,3,3,2,1] -> [2,2,2,2,1] 3->2, 3 ops
[2,2,2,2,1] -> [1,1,1,1,1] 2->1, 4 ops
total = 1 + 2 + 3 + 4 = 10

Sort the array in reverse order, 
if we find a number at index i that is is smaller than the previous number, 
we need i ops to make all the numbers before it to become itself.

Time complexity: O(nlogn)
Space complexity: O(1)*/
public class Solution {
    public int ReductionOperations(int[] nums) {
        Array.Sort(nums, (a,b) => b - a);
    int ans = 0;
    for (int i = 1; i < nums.Length; ++i)
      if (nums[i] != nums[i - 1]) ans += i;    
    return ans;
    }
}

// 1888. Minimum Number of Flips to Make the Binary String Alternating
/*Solution: Sliding Window
Trying all possible rotations will take O(n2) that leads to TLE, we have to do better.

concatenate the s to itself, then using a sliding window length of n to check how many count needed to make the string in the window alternating which will cover all possible rotations. We can update the count in O(1) when moving to the next window.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MinFlips(string s) {
        int n = s.Length;    
    int ans = Int32.MaxValue;
    for (int i = 0, c0 = 0, c1 = 1, a0 = 0, a1 = 0; i < 2 * n; ++i, c0 ^= 1, c1 ^= 1) {
      if (s[i % n] - '0' != c0) ++a0;
      if (s[i % n] - '0' != c1) ++a1;
      if (i < n - 1) continue;
      if (i >= n) {
        if ((s[i - n] - '0') != (c0 ^ (n & 1))) --a0;
        if ((s[i - n] - '0') != (c1 ^ (n & 1))) --a1;
      }
      ans = Math.Min(Math.Min(ans, a0), a1);      
    }    
    return ans;
    }
}

// 1889. Minimum Space Wasted From Packaging
/*Solution: Greedy + Binary Search
sort packages and boxes
for each box find all (unpacked) packages that are smaller or equal to itself.
Time complexity: O(nlogn) + O(mlogm) + O(mlogn)
Space complexity: O(1)*/
public class Solution {
    public int MinWastedSpace(int[] packages, int[][] boxes) {
     int kMod = (int)1e9 + 7;
    int n = packages.Length;        
    Array.Sort(packages);   
    // int bit = 0;
   //  int eit = packages.Length - 1;
    long sum = 0L; Array.ForEach(packages, i => sum += i);
    long ans = long.MaxValue;
    foreach (int[] box in boxes) {
      Array.Sort(box);
      int l = 0;
      long cur = 0;
      foreach (long b in box) {
        int r = upperBound(packages, (int)b); //- bit;
        cur += b * (r - l);
        if (r == n) {
          ans = Math.Min(ans, cur - sum);
          break;
        }
        l = r;
      }      
    }
    return ans == long.MaxValue ? -1 : Convert.ToInt32(ans % kMod);
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

// 1884. Egg Drop With 2 Eggs and N Floors
/*Solution: Math
Time complexity: O(1)
Space complexity: O(1)

*/
public class Solution {
    public int TwoEggDrop(int n) {
        int ans = 0;
    while (ans < n) n -= ans++;
    return ans;
    }
}

// 1883. Minimum Skips to Arrive at Meeting On Time
/*Solution: DP
Let dp[i][k] denote min (time*speed) to finish the i-th road with k rest.

dp[i][k] = min(dp[i – 1][k – 1] + dist[i] / speed * speed, # skip the rest,
(dp[i-1][k] + dist[i] + speed – 1) // speed * speed # rest

ans = argmin(dp[n][k] <= hours * speed)

Time complexity: O(n2)
Space complexity: O(n2)

*/
public class Solution {
    public int MinSkips(int[] dist, int speed, int hoursBefore) {
        int n = dist.Length;
    long total = 0L; Array.ForEach(dist, i => total += i);
    if (total > (long)(speed) * hoursBefore) return -1;
    long[][] dp = new long[n + 1][]; 
    for(int i = 0; i < n + 1; i++) 
    {dp[i] = new long[n + 1];
     Array.Fill(dp[i], long.MaxValue / 2);}
       // (n + 1, vector<long>(n + 1, Long.MaxValue / 2));
    dp[0][0] = 0;
    for (int i = 1; i <= n; ++i)
      for (int k = 0; k <= i; ++k)
        dp[i][k] = Math.Min((dp[i - 1][k] + dist[i - 1] + speed - 1) / speed * speed,
                       (k > 0 ? dp[i - 1][k - 1] + dist[i - 1] : long.MaxValue));
    for (int k = 0; k < n; ++k)
      if (dp[n][k] <= (long)(speed) * hoursBefore) return k;
    return -1;
    }
}

// 1882. Process Tasks Using Servers
/*Solution: Simulation / Priority Queue
Two priority queues, one for free servers, another for releasing events.
One FIFO queue for tasks to schedule.

Time complexity: O(nlogn)
Space complexity: O(n)*/
public class Solution {
    public int[] AssignTasks(int[] servers, int[] tasks) {
        int n = servers.Length;
    int m = tasks.Length;
    //using P = pair<long, int>;
   // priority_queue<P, vector<P>, greater<P>> frees, release;
        PriorityQueue<KeyValuePair<int,int>, KeyValuePair<int,int>> frees = new PriorityQueue<KeyValuePair<int,int>, KeyValuePair<int,int>>(Comparer<KeyValuePair<int,int>>.Create((a, b) => a.Key == b.Key?a.Value.CompareTo(b.Value) : a.Key.CompareTo(b.Key)));
        PriorityQueue<KeyValuePair<int,int>, KeyValuePair<int,int>> release = new PriorityQueue<KeyValuePair<int,int>, KeyValuePair<int,int>>(Comparer<KeyValuePair<int,int>>.Create((a, b) => a.Key == b.Key?a.Value.CompareTo(b.Value) : a.Key.CompareTo(b.Key)));
    for (int i = 0; i < n; ++i)
      frees.Enqueue(new KeyValuePair<int,int>(servers[i], i), new KeyValuePair<int,int>(servers[i], i));    
    int[] ans = new int[m];
    Queue<int> q = new Queue<int>();
    int l = 0;
   int t = 0;
    int count = 0;
    while (count != m) {      
      // Release servers.      
      while (release.Count != 0 && release.Peek().Key <= t) {
        var (rt, i) = release.Peek(); release.Dequeue();
        frees.Enqueue(new KeyValuePair<int,int>(servers[i], i),new KeyValuePair<int,int>(servers[i], i));        
      }
      // Enqueue tasks.
      while (l < m && l <= t) q.Enqueue(l++);
      // Schedule tasks.
      while (q.Count > 0 && frees.Count > 0) {
        int j = q.Peek(); q.Dequeue();
        var (w, i) = frees.Peek(); frees.Dequeue();
        release.Enqueue(new KeyValuePair<int,int>(t + tasks[j], i),new KeyValuePair<int,int>(t + tasks[j], i));
        ans[j] = i;
        ++count;
      }
      // Advance time.
      if (frees.Count == 0 && release.Count != 0) {             
        t = release.Peek().Key;
      } else {
        ++t;
      }
    }
    return ans;
    }
}
//tuple
public class Solution {
    public int[] AssignTasks(int[] servers, int[] tasks) {
        int n = servers.Length;
    int m = tasks.Length;
    //using P = pair<long, int>;
   // priority_queue<P, vector<P>, greater<P>> frees, release;
        PriorityQueue<(int a0,int a1), (int a0,int a1)> frees = new PriorityQueue<(int ,int ), (int ,int )>(Comparer<(int a0,int a1)>.Create((a, b) => a.a0 == b.a0? a.a1 - b.a1 : a.a0 - b.a0));
    PriorityQueue<(int a0,int a1), (int a0,int a1)> release = new PriorityQueue<(int ,int ), (int ,int )>(Comparer<(int a0,int a1)>.Create((a, b) => a.a0 == b.a0? a.a1 - b.a1 : a.a0 - b.a0));
    for (int i = 0; i < n; ++i)
      frees.Enqueue((servers[i], i), (servers[i], i));    
    int[] ans = new int[m];
    Queue<int> q = new  Queue<int>();
    int l = 0;
    int t = 0;
    int count = 0;
    while (count != m) {      
      // Release servers.      
      while (release.Count != 0 && release.Peek().a0 <= t) {
        var (rt, i) = release.Peek(); release.Dequeue();
        frees.Enqueue((servers[i], i),(servers[i], i));        
      }
      // Enqueue tasks.
      while (l < m && l <= t) q.Enqueue(l++);
      // Schedule tasks.
      while (q.Count > 0 && frees.Count > 0) {
        int j = q.Peek(); q.Dequeue();
        var (w, i) = frees.Peek(); frees.Dequeue();
        release.Enqueue((t + tasks[j], i),(t + tasks[j], i));
        ans[j] = i;
        ++count;
      }
      // Advance time.
      if (frees.Count == 0 && release.Count != 0) {             
        t = release.Peek().a0;
      } else {
        ++t;
      }
    }
    return ans;
    }
}

// 1881. Maximum Value after Insertion
/*Solution: Greedy
Find the best position to insert x. 
For positive numbers, 
insert x to the first position i such that s[i] < x or s[i] > x for negatives.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public string MaxValue(string n, int x) {
        int l = n.Length;    
    if (n[0] == '-') {
      for (int i = 1; i <= l; ++i)
        if (i == l || n[i] - '0' > x)
          return n.Substring(0, i) + (char)('0' + x) + n.Substring(i);
    } else {
      for (int i = 0; i <= l; ++i)
        if (i == l || x > n[i] - '0')
          return n.Substring(0, i) + (char)('0' + x) + n.Substring(i);
    }
    return "";
    }
}

// 1880. Check if Word Equals Summation of Two Words
/*Solution: Brute Force
Tips: Write a reusable function to compute the score of a word.

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public bool IsSumEqual(string firstWord, string secondWord, string targetWord) {
        int score(string s) {
      int ans = 0;
      foreach (char c in s)
        ans = ans * 10 + (c - 'a');
      return ans;
    };
    return score(firstWord) + score(secondWord) == score(targetWord);
    }
}

// 1879. Minimum XOR Sum of Two Arrays
/*Solution: DP / Permutation to combination
dp[s] := min xor sum by using a subset of nums2 
(presented by a binary string s) xor with nums1[0:|s|].

Time complexity: O(n*2n)
Space complexity: O(2n)*/
using System.Numerics;
public class Solution {
    public int MinimumXORSum(int[] nums1, int[] nums2) {
         int n = nums1.Length;
    int[] dp = new int[1 << n];Array.Fill(dp, Int32.MaxValue);
    dp[0] = 0;
    for (int s = 0; s < 1 << n; ++s) {
      int index = BitOperations.PopCount((uint)s);
      for (int i = 0; i < n; ++i) {
        if ((s & (1 << i) )> 0) continue;
        dp[s | (1 << i)] =Math.Min(dp[s | (1 << i)],
                               dp[s] + (nums1[index] ^ nums2[i]));
      }
    }    
    return dp[dp.Length - 1];
    }
}

// 1878. Get Biggest Three Rhombus Sums in a Grid
/*Solution: Brute Force
Just find all Rhombus…

Time complexity: O(mn*min(n,m)2)
Space complexity: O(mn*min(n,m)2)*/
public class Solution {
    public int[] GetBiggestThree(int[][] grid) {
        int m = grid.Length;
    int n = grid[0].Length;
    List<int> ans = new List<int>();
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        ans.Add(grid[i][j]);
    for (int a = 2; a <= Math.Min(m, n); ++a)
      for (int cy = a - 1; cy + a <= m; ++cy)
        for (int cx = a - 1; cx + a <= n; ++cx) {
          int s = grid[cy][cx - a + 1]
                + grid[cy][cx + a - 1]
                + grid[cy + a - 1][cx]
                + grid[cy - a + 1][cx];
          for (int i = 1; i < a - 1; ++i)
            s += grid[cy - i][cx - a + i + 1] 
               + grid[cy - i][cx + a - i - 1]
               + grid[cy + i][cx - a + i + 1] 
               + grid[cy + i][cx + a - i - 1];
          ans.Add(s);          
        }
   ans.Sort((a, b) => b - a);
    List<int> output = new List<int>();
    
    foreach (int x in ans) {
      if (output.Count == 0 || output[output.Count - 1] != x)
        output.Add(x);
      if (output.Count == 3) break;
    }
    return output.ToArray();
    }
}

// 1877. Minimize Maximum Pair Sum in Array
/*Solution: Greedy
Sort the elements, pair nums[i] with nums[n – i – 1] and find the max pair.

Time complexity: O(nlogn) -> O(n) counting sort.
Space complexity: O(1)*/
public class Solution {
    public int MinPairSum(int[] nums) {
        int n = nums.Length;
    int ans = 0;
    Array.Sort(nums);
    for (int i = 0; i < n / 2; ++i)
      ans = Math.Max(ans, nums[i] + nums[n - i - 1]);
    return ans;
    }
}

// 1876. Substrings of Size Three with Distinct Characters
/*Solution: Brute Force w/ (Hash)Set
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int CountGoodSubstrings(string s) {
        int ans = 0;
    for (int i = 0; i + 2 < s.Length; ++i) 
      ans += Convert.ToInt32(s.Substring(i, 3).ToHashSet().Count == 3);
    return ans;
    }
}

// 1872. Stone Game VIII
/*Note: Naive DP (min-max) takes O(n2) which leads to TLE. 
The key of this problem is that each player takes k stones, 
but put their sum back as a new stone, 
so you can assume all the original stones are still there, 
but opponent has to start from the k+1 th stone.

Let dp[i] denote the max score diff that current player can achieve 
by taking stones[0~i] (or equivalent)

dp[n-1] = sum(A[0~n-1]) 
// Alice takes all the stones.
dp[n-2] = sum(A[0~n-2]) – (A[n-1] + sum(A[0~n-2])) = sum(A[0~n-2]) – dp[n-1] 
// Alice takes n-1 stones, Bob take the last one (A[n-1]) + put-back-stone.
dp[n-3] = sum(A[0~n-3]) – max(dp[n-2], dp[n-1]) 
// Alice takes n-2 stones, Bob has two options (takes n-1 stones or takes n stones)
…
dp[0] = A[0] – max(dp[n-1], dp[n-1], …, dp[1]) 
// Alice takes the first stone, Bob has n-1 options.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int StoneGameVIII(int[] stones) {
        int n = stones.Length;
    for (int i = 1; i < n; ++i)
      stones[i] += stones[i - 1];
    int ans = stones[stones.Length - 1]; // take all the stones.
    for (int i = n - 2; i > 0; --i)
      ans = Math.Max(ans, stones[i] - ans);
    return ans;
    }
}

// 1871. Jump Game VII
/*One Pass DP

Explanation
dp[i] = true if we can reach s[i].
pre means the number of previous position that we can jump from.

Complexity
Time O(n)
Space O(n)*/
public class Solution {
    public bool CanReach(string s, int minJump, int maxJump) {
       int n = s.Length, pre = 0;
        bool[] dp = new bool[n];
        dp[0] = true;
        for (int i = 1; i < n; ++i) {
            if (i >= minJump && dp[i - minJump])
                pre++;
            if (i > maxJump && dp[i - maxJump - 1])
                pre--;
            dp[i] = pre > 0 && s[i] == '0';
        }
        return dp[n - 1];
    }
}

// 1869. Longer Contiguous Segments of Ones than Zeros
/*Solution: Brute Force
Write a function count to count longest contiguous segment of m, 
return count(‘1’) > count(‘0’)

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public bool CheckZeroOnes(string s) {
        int count(char m) {
      int ans = 0;
      int l = 0;
      foreach (char c in s)
        if (c != m) l = 0;
        else ans = Math.Max(ans, ++l);      
      return ans;
    };
    return count('1') > count('0');
    }
}

// 1866. Number of Ways to Rearrange Sticks With K Sticks Visible
/*Solution: DP
dp(n, k) = dp(n – 1, k – 1) + (n-1) * dp(n-1, k)

Time complexity: O(n*k)
Space complexity: O(n*k) -> O(k)

*/
public class Solution {
    public int RearrangeSticks(int n, int k) {
         int kMod = (int)1e9 + 7;
    long[][] dp = new long[n + 1][];
        //, vector<long>(k + 1));    
    for(int i = 0; i < n+1; i++) dp[i] = new long[k + 1];
    for (int j = 1; j <= k; ++j) {
      dp[j][j] = 1;
      for (int i = j + 1; i <= n; ++i)
        dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j] * (i - 1)) % kMod;
    }
    return (int)dp[n][k];
    }
    
}

// 1865. Finding Pairs With a Certain Sum
/*Solution: HashTable
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

Space complexity: O(m + n)*/
public class FindSumPairs {

    private int[] nums1_ ;
    private int[] nums2_ ;
    private Dictionary<int, int> freq_; // = new Dictionary<int, int>();
    public FindSumPairs(int[] nums1, int[] nums2) {
         nums1_ =  nums1; nums2_ = nums2;
        freq_ = new Dictionary<int, int>();
         foreach (int x in nums2_) freq_[x] = freq_.GetValueOrDefault(x,0)+1;
        
    }
    
    public void Add(int index, int val) {
        freq_[nums2_[index]] = freq_.GetValueOrDefault(nums2_[index],0)-1;
        if (freq_[nums2_[index]] == 0) freq_.Remove(nums2_[index]);
       
        nums2_[index] += val;
         freq_[nums2_[index]] = freq_.GetValueOrDefault(nums2_[index],0)+1;
     
    }
    
    public int Count(int tot) {
        int ans = 0;
    foreach (int a in nums1_) {
        if(freq_.ContainsKey(tot - a)){
            ans += freq_[tot - a];
        } 
        //ans += freq_.GetValueOrDefault(tot - a, 0);
    }
    return ans;
    }
}

/**
 * Your FindSumPairs object will be instantiated and called as such:
 * FindSumPairs obj = new FindSumPairs(nums1, nums2);
 * obj.Add(index,val);
 * int param_2 = obj.Count(tot);
 */

 public class FindSumPairs {

    private int[] nums1_ ;
    private int[] nums2_ ;
    private Dictionary<int, int> freq_; // = new Dictionary<int, int>();
    public FindSumPairs(int[] nums1, int[] nums2) {
         nums1_ =  nums1; nums2_ = nums2;
        freq_ = new Dictionary<int, int>();
         foreach (int x in nums2_) freq_[x] = freq_.GetValueOrDefault(x,0)+1;
        
    }
    
    public void Add(int index, int val) {
        freq_[nums2_[index]] = freq_.GetValueOrDefault(nums2_[index],0)-1;
        if (freq_[nums2_[index]] == 0) freq_.Remove(nums2_[index]);
       
        nums2_[index] += val;
         freq_[nums2_[index]] = freq_.GetValueOrDefault(nums2_[index],0)+1;
     
    }
    
    public int Count(int tot) {
        int ans = 0;
    foreach (int a in nums1_) {
        /*if(freq_.ContainsKey(tot - a)){
            ans += freq_[tot - a];
        } */
        ans += freq_.GetValueOrDefault(tot - a, 0); // => slow
    }
    return ans;
    }
}

/**
 * Your FindSumPairs object will be instantiated and called as such:
 * FindSumPairs obj = new FindSumPairs(nums1, nums2);
 * obj.Add(index,val);
 * int param_2 = obj.Count(tot);
 */

// 1864. Minimum Number of Swaps to Make the Binary String Alternating
/*Solution: Greedy
Two passes, make the string starts with ‘0’ or ‘1’, 
count how many 0/1 swaps needed. 
0/1 swaps must equal otherwise it’s impossible to swap.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MinSwaps(string s) {
        int count(int m) {
      int[] swaps = new int[2];
      for (int i = 0; i < s.Length; ++i, m ^= 1)
        if (s[i] - '0' != m)
          ++swaps[s[i] - '0'];
      return swaps[0] == swaps[1] ? swaps[0] : Int32.MaxValue;
    };
    
    int ans = Math.Min(count(0), count(1));
    return ans != Int32.MaxValue ? ans : -1;
    }
}

// 1863. Sum of All Subset XOR Totals

// 1860. Incremental Memory Leak

// 1859. Sorting the Sentence

// 1857. Largest Color Value in a Directed Graph

// 1854. Maximum Population Year
/*Solution: Simulation
Time complexity: O(n*y)
Space complexity: O(y)*/
public class Solution {
    public int MaximumPopulation(int[][] logs) {
        int[] pop = new int[2051];
    foreach (int[] log in logs) {
      for (int y = log[0]; y < log[1]; ++y)
        ++pop[y];
    }
    int ans = -1;
    int max_pop = 0;
    for (int y = 1950; y <= 2050; ++y) {
      if (pop[y] > max_pop) {
        max_pop = pop[y];
        ans = y;
      }
    }
    return ans;
    }
}

// 1851. Minimum Interval to Include Each Query
/*Solution: Offline Processing + Priority Queue
Similar to 花花酱 LeetCode 1847. Closest Room

Sort intervals by right in descending order, 
sort queries in descending. 
Add valid intervals into the priority queue (or treeset) 
ordered by size in ascending order. 
Erase invalid ones. 
The first one (if any) will be the one with the smallest size 
that contains the current query.

Time complexity: O(nlogn + mlogm + mlogn)
Space complexity: O(m + n)

*/
public class Solution {
    public int[] MinInterval(int[][] intervals, int[] queries) {
        int n = intervals.Length;
    int m = queries.Length;    
    Array.Sort(intervals , (a, b) =>  b[1] - a[1]);
    KeyValuePair<int, int>[] qs = new KeyValuePair<int, int>[m]; // {query, i}
    for (int i = 0; i < m; ++i)
      qs[i] = new KeyValuePair<int,int>(queries[i], i);
    Array.Sort(qs, (a,b) => b.Key - a.Key );
 
    int[] ans = new int[m];
    int j = 0;
    PriorityQueue<KeyValuePair<int,int>, KeyValuePair<int,int>> pq = new PriorityQueue<KeyValuePair<int,int>, KeyValuePair<int,int>>(Comparer<KeyValuePair<int,int>>.Create((a, b) => b.Key == a.Key ? b.Value - a.Value : b.Key - a.Key));
    
    foreach (var (query, i) in qs) {
      while (j < n && intervals[j][1] >= query) {
        pq.Enqueue(new KeyValuePair<int,int>(-(intervals[j][1] - intervals[j][0] + 1), intervals[j][0]), new KeyValuePair<int,int>(-(intervals[j][1] - intervals[j][0] + 1), intervals[j][0]));
        ++j;
      }
      while (pq.Count != 0 && pq.Peek().Value > query) 
        pq.Dequeue();      
      ans[i] = pq.Count == 0 ? -1 : -pq.Peek().Key;         
    }
    return ans;
    }
}

// 1850. Minimum Adjacent Swaps to Reach the Kth Smallest Number
/*With the helper method from 31. next permutation.
https://leetcode.com/problems/next-permutation/
we can use brute force to calculate the minimal 
adjacent swap to make 2 arrays equal

Solution: Next Permutation + Greedy
Time complexity: O(k*n + n^2)
Space complexity: O(n)

*/

public class Solution {
    public int GetMinSwaps(string num, int k) {
       int N = num.Length;
        int[] origin = new int[num.Length];
        int[] nums = new int[num.Length];
        for (int i = 0; i < num.Length; i++) {
            nums[i] = num[i] - '0';
            origin[i] = num[i] - '0';
        }
        while (k-- > 0) nextPermutation(nums);
        int res = 0;
        for (int i = 0; i < N; i++)
            if (nums[i] != origin[i]) {
                int j = i;
                while (nums[j] != origin[i]) j++;
                for (int x = j; x > i; x--) {
                    swap(nums, x, x - 1);
                    res++;
                }
            }
        return res;
    }
    
     public void nextPermutation(int[] nums) {
        if (nums.Length <= 1) return;
        int i = nums.Length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) i--; 
        if (i >= 0) {
            int j = nums.Length - 1;
            while (j > i && nums[j] <= nums[i]) j--; 
            swap(nums, i, j);
        }
        reverse(nums, i + 1, nums.Length - 1);
    }
    private void reverse(int[] nums, int left, int right) {
        while (left < right)
            swap(nums, left++, right--);
    }
    private void swap (int[] nums, int i, int j) {
        int tmp = nums[j];
        nums[j] = nums[i];
        nums[i] = tmp;
    }
    
}

// 1849. Splitting a String Into Descending Consecutive Values
/*Solution: DFS
Time complexity: O(2n)
Space complexity: O(n)

*/
public class Solution {
    public bool SplitString(string s) {
        int n = s.Length;
    List<long> nums = new List<long>();
    bool dfs (int p) {
      if (p == n) return nums.Count >= 2;
      long cur = 0;
      for (int i = p; i < n && cur < 1e11; ++i) {        
        cur = cur * 10 + (s[i] - '0');
        if (nums.Count == 0 || cur + 1 == nums[nums.Count - 1]) {
          nums.Add(cur);
          if (dfs(i + 1)) return true;
          nums.RemoveAt(nums.Count - 1);
        }
        if (nums.Count != 0 && cur >= nums[nums.Count - 1]) break;                
      }
      return false;
    };
    return dfs(0);
    }
}

// 1848. Minimum Distance to the Target Element
/*Solution: Brute Force
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int GetMinDistance(int[] nums, int target, int start) {
        int ans = Int32.MaxValue;
    for (int i = 0; i < nums.Length; ++i)
      if (nums[i] == target)
        ans = Math.Min(ans, Math.Abs(i - start));    
    return ans;
    }
}

// 1847. Closest Room
/*Solution 2: Fenwick Tree

Time complexity: O(nlogS + mlogSlogn)
Space complexity: O(nlogS)*/
public class Solution {
    public int[] ClosestRoom(int[][] rooms, int[][] queries) {
   S = 0;
    foreach (var r in rooms) S = Math.Max(S, r[1]); 
    foreach (var r in rooms) update(r[1], r[0]);    
    List<int> ans = new List<int>();
    foreach (var q in queries)
      ans.Add(query(q[1], q[0]));
    return ans.ToArray();
  }
private int S;
 public void update(int s, int id) {
    for (s = S - s + 1; s <= S; s += s & (-s)){
        tree[s] = tree.GetValueOrDefault(s, new List<int>() );
     tree[s].Add(id);
    }
      
  }
  
 public int query(int s, int id) {
    int ans = -1;
    for (s = S - s + 1; s > 0; s -= s & (-s)) {
      if (!tree.ContainsKey(s)) continue;
      int id1 = Int32.MaxValue;      
      int id2 = Int32.MaxValue;
        tree[s].Sort();
      int it = lowerBound( tree[s], id);
      if (it !=  tree[s].Count) id1 =  tree[s][it];      
      if (it > 0) id2 =  tree[s][it - 1];      
      int cid = (Math.Abs(id1 - id) < Math.Abs(id2 - id)) ? id1 : id2;
      if (ans == -1 || Math.Abs(cid - id) < Math.Abs(ans - id))
        ans = cid;
      else if ( Math.Abs(cid - id) == Math.Abs(ans - id))
        ans = Math.Min(ans, cid);
    }
    return ans;
  }
  private Dictionary<int,List<int>> tree = new Dictionary<int,List<int>>();
    
     public int lowerBound(List<int> nums, int target ) {
        if(nums == null) {
            return -1;
        }
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
}
// Tuple Array
public class Solution {
    public int[] ClosestRoom(int[][] rooms, int[][] queries) {
    var sorted = new (int size, int id)[rooms.Length];
        for (int i = 0; i < rooms.Length; i++)
            sorted[i] = (rooms[i][1], rooms[i][0]);
        Array.Sort(sorted);
        
        var result = new int[queries.Length];
        for (int i = 0; i < queries.Length; i++)
        {
            var minIndex = GetMinIndex(sorted, queries[i][1]);
            if (minIndex == -1)
            {
                result[i] = -1;
                continue;
            }   
            
            var closestRoom = sorted[minIndex].id;
            var minDiff = Math.Abs(queries[i][0] - closestRoom);
            for (int j = minIndex + 1; j < sorted.Length; j++)
            {
                var currDiff = Math.Abs(queries[i][0] - sorted[j].id);
                if (currDiff < minDiff || (currDiff == minDiff && sorted[j].id < closestRoom))
                {
                    closestRoom = sorted[j].id;
                    minDiff = Math.Abs(queries[i][0] - closestRoom);
                }
            }
            result[i] = closestRoom;
        }            
        return result;
    }
    
    private int GetMinIndex((int size, int id)[] rooms, int min)
    {       
        var result = -1;
        var left = 0;
        var right = rooms.Length - 1;
        while (left <= right)
        {
            var mid = left + (right - left) / 2;
            if (rooms[mid].size < min)
                left = mid + 1;
            else
            {
                result = mid;
                right = mid - 1;
            }
        }
        return result;
    }
}
// 42 / 42 test cases passed, but took too long.
/*Solution: Off Processing: Sort + Binary Search

Time complexity: O(nlogn + mlogm)
Space complexity: O(n + m)

Sort queries and rooms by size in descending order, 
only add valid rooms (size >= min_size) to the treeset for binary search.

*/
public class Solution {
    public int[] ClosestRoom(int[][] rooms, int[][] queries) {
   int n = rooms.Length;
     int m = queries.Length;
        
    for (int i = 0; i < m; ++i){
      queries[i] =  new int[] { queries[i][0], queries[i][1], i};
    }
        
    // Sort queries by size DESC.
    Array.Sort( queries, (q1, q2) => q2[1] - q1[1]);
    
    // Sort room by size DESC.
    Array.Sort(rooms, (r1, r2) => r2[1] - r1[1]);

    int[] ans = new int [m]; 
    int j = 0;
    List<int> ids = new List<int>();
    foreach (var q in queries) {
      while (j < n && rooms[j][1] >= q[1])
        ids.Add(rooms[j++][0]);      
      if (ids.Count == 0) {
        ans[q[2]] = -1;
        continue;
      }
        ids.Sort();
      int id = q[0]; //List<int> ds = ids.ToList();
       // Console.WriteLine("ds:  " + ds.ToString());
      int it = lowerBound(ids, id );
      int id1 = (it != ids.Count) ? ids[it] : Int32.MaxValue;
      int id2 = (it > 0)? ids[it - 1]: id1;
     // if (it > 0) id2 = ids[it - 1];
      ans[q[2]] = Math.Abs(id1 - id) < Math.Abs(id2 - id) ? id1 : id2;
    }
    return ans;
}
    
    public int lowerBound(List<int> nums, int target ) {
        if(nums == null) {
            return -1;
        }
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
}

// 1846. Maximum Element After Decreasing and Rearranging
/*Solution: Sort
arr[0] = 1,
arr[i] = min(arr[i], arr[i – 1] + 1)

ans = arr[n – 1]

Time complexity: O(nlogn)
Space complexity: O(1)*/
public class Solution {
    public int MaximumElementAfterDecrementingAndRearranging(int[] arr) {
        int n = arr.Length;
    Array.Sort(arr);
    arr[0] = 1;
    for (int i = 1; i < n; ++i)
      arr[i] = Math.Min(arr[i], arr[i - 1] + 1);
    return arr[arr.Length - 1];
    }
}

// 1844. Replace All Digits with Characters
/*s[i] - '0' change a character to integer.
shift(c, x) is (char)(c + x) in java.

Solution: Simulation
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public string ReplaceDigits(string s) {
        char[] ans = s.ToCharArray();
        for (int i = 1; i < s.Length; i += 2)
      ans[i] += (char)(s[i - 1] - '0');
    return new String(ans);
    }
}

// 1840. Maximum Building Height
/*Solution: Two Passes
Trim the max heights based on neighboring max heights.
Two passes: left to right, right to left.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MaxBuilding(int n, int[][] restrictions) {
        List<int[]>rs =  restrictions.ToList();
        rs.Add( new int[2]{1, 0});
    rs.Sort((a,b) => a[0] - b[0]);
    if (rs[rs.Count - 1][0] != n)
      rs.Add( new int[2]{n, n - 1});
    
    int m = rs.Count;
    for (int i = 1; i < m; ++i)
      rs[i][1] =  Math.Min(rs[i][1], rs[i - 1][1] + rs[i][0] - rs[i - 1][0]);
    for (int i = m - 2; i >= 0; --i)
      rs[i][1] =  Math.Min(rs[i][1], rs[i + 1][1] - rs[i][0] + rs[i + 1][0]);
    
    int ans = 0;
    for (int i = 1; i < m; ++i) {
      int l = rs[i - 1][1];
      int r = rs[i][1];
      ans =  Math.Max(ans,  Math.Max(l, r) + (rs[i][0] - rs[i - 1][0] - Math.Abs(l - r)) / 2);
    }
    return ans;
    }
}

// 1839. Longest Substring Of All Vowels in Order
/*Solution: Counter
Use a counter to track how many unique vowels we saw so far. 
Reset the counter whenever the s[i] < s[i-1]. 
Incase the counter if s[i] > s[i – 1]. 
When counter becomes 5, we know we found a valid substring.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int LongestBeautifulSubstring(string word) {
        int n = word.Length;
    int ans = 0;
    char p = (char)('a' - 1);
    for (int i = 0, vowels = 0, l = 0; i < n; ++i) {
      if (word[i] < p) {
        vowels = Convert.ToInt32(word[i] == 'a');
        l = i;
      } else if (word[i] > p) {
        ++vowels;
      }      
      if (vowels == 5)
        ans = Math.Max(ans, i - l + 1);
      p = word[i];
    }
    return ans;
    }
}

// 1837. Sum of Digits in Base K
/*Solution: Base Conversion
Time complexity: O(logn)
Space complexity: O(1)

*/
public class Solution {
    public int SumBase(int n, int k) {
        int ans = 0;
    while (n > 0) {
      ans += n % k;
      n /= k;
    }
    return ans;
    }
}

// 1835. Find XOR Sum of All Pairs Bitwise AND
/*Solution: Bit
(a[0] & b[i]) ^ (a[1] & b[i]) ^ … ^ (a[n-1] & b[i]) = (a[0] ^ a[1] ^ … ^ a[n-1]) & b[i]

We can pre compute that xor sum of array A.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int GetXORSum(int[] arr1, int[] arr2) {
        int ans = 0;    
    int xora = 0;
    foreach (int a in arr1) xora ^= a;    
    foreach (int b in arr2) ans ^= (xora & b);
    return ans;
    }
}

// 1834. Single-Threaded CPU
/*Solution: Simulation w/ Sort + PQ
Time complexity: O(nlogn)
Space complexity: O(n)

*/
public class Solution {
    public int[] GetOrder(int[][] tasks) {
         int n = tasks.Length;
    for (int j = 0; j < n; ++j){
         tasks[j] = new int[3]{ tasks[j][0], tasks[j][1], j};
    }
     
    Array.Sort(tasks, (a,b) => a[0] - b[0]); // sort by enqueue_time;    
    
    List<int> ans = new List<int>();
     PriorityQueue<KeyValuePair<int,int>, KeyValuePair<int,int>> q = new PriorityQueue<KeyValuePair<int,int>, KeyValuePair<int,int>>(Comparer<KeyValuePair<int,int>>.Create((a, b) => b.Key == a.Key ? b.Value - a.Value : b.Key - a.Key));
    // {-processing_time, -index}
    int i = 0;
    long t = 0;    
    while (ans.Count != n) {
      // Advance to next enqueue time if q is empty.
      if (i < n && q.Count == 0 && tasks[i][0] > t)
        t = tasks[i][0];
      // Enqueue all available tasks.
      while (i < n && tasks[i][0] <= t) {
        q.Enqueue(new KeyValuePair<int,int>(-tasks[i][1], -tasks[i][2]), new KeyValuePair<int,int>(-tasks[i][1], -tasks[i][2]));
        ++i;
      }
      // Extra the top one.
      t -= q.Peek().Key;
      ans.Add(-q.Peek().Value);
      q.Dequeue();      
    }
    return ans.ToArray();
    }
}

public class Solution {
    public int[] GetOrder(int[][] tasks) {
        int n = tasks.Length;
    for (int j = 0; j < n; ++j){
         tasks[j] = new int[3]{ tasks[j][0], tasks[j][1], j};
    }
     
    Array.Sort(tasks, (a,b) => a[0] - b[0]); // sort by enqueue_time;    
    
    List<int> ans = new List<int>();
    PriorityQueue<int[], int[]> q = new PriorityQueue<int[], int[]>(Comparer<int[]>.Create((a, b) => a[0] == b[0] ? b[1] - a[1] : b[0] - a[0]));
    // {-processing_time, -index}
    int i = 0;
    long t = 0;    
    while (ans.Count != n) {
      // Advance to next enqueue time if q is empty.
      if (i < n && q.Count == 0 && tasks[i][0] > t)
        t = tasks[i][0];
      // Enqueue all available tasks.
      while (i < n && tasks[i][0] <= t) {
        q.Enqueue(new int[2]{-tasks[i][1], -tasks[i][2]}, new int[2]{-tasks[i][1], -tasks[i][2]});
        ++i;
      }
      // Extra the top one.
      t -= q.Peek()[0];
      ans.Add(-q.Peek()[1]);
      q.Dequeue();      
    }
    return ans.ToArray();
    }
}
// 1833. Maximum Ice Cream Bars
/*Solution: Greedy
Sort by price in ascending order, buy from the lowest price to the highest price.

Time complexity: O(nlogn)
Space complexity: O(1)

*/
public class Solution {
    public int MaxIceCream(int[] costs, int coins) {
       Array.Sort(costs);
    int ans = 0;
    foreach (int c in costs) {
      if (c > coins) break;
      coins -= c;
      ++ans;
    }
    return ans;
    }
}

// 1832. Check if the Sentence Is Pangram
/*Solution: Hashset
Time complexity: O(n)
Space complexity: O(26)*/
public class Solution {
    public bool CheckIfPangram(string sentence) {
         HashSet<char> s = new HashSet<char>();
    foreach (char c in sentence)
      s.Add(c);
    return s.Count == 26;
    }
}

// 1830. Minimum Number of Operations to Make String Sorted
/*Solution: Math
Time complexity: O(26n)
Space complexity: O(n)

*/
public class Solution {
    public int kMod = (int)1e9 + 7;
    public Int64 powm(Int64 b, Int64 p) {
  Int64 ans = 1;
  while (p > 0) {
    if ((p & 1) > 0 ) ans = (ans * b) % kMod;
    b = (b * b) % kMod;
    p >>= 1;
  }
  return ans;
}
    public int MakeStringSorted(string s) {
        int n = s.Length;    
    Int64[] fact = new Int64[n + 1]; Array.Fill(fact, 1);
    Int64[] inv = new Int64[n + 1]; Array.Fill(inv, 1);
    
    for (int i = 1; i <= n; ++i) {
      fact[i] = (fact[i - 1] * i) % kMod;
      inv[i] = powm(fact[i], kMod - 2);
    }
    
    Int64 ans = 0;
    int[] freq = new int[26];
    for (int i = n - 1; i >= 0; --i) {
      int idx = s[i] - 'a';
      ++freq[idx];
        Int64 sum = 0;
        for (int a = 0; a < idx; a++)
            sum += freq[a];
      Int64 cur =  sum * fact[n - i - 1] % kMod;
      foreach (int f in freq)
        cur = cur * inv[f] % kMod;      
      ans = (ans + cur) % kMod;
    }
    return (int)(ans);
    }
}

// 1829. Maximum XOR for Each Query
/*Solution: Prefix XOR
Compute s = nums[0] ^ nums[1] ^ … nums[n-1] first

to remove nums[i], we just need to do s ^= nums[i]

We can always maximize the xor of s and k to (2^maxbit – 1)
k = (2 ^ maxbit – 1) ^ s

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int[] GetMaximumXor(int[] nums, int maximumBit) {
        int t = (1 << maximumBit) - 1;
    int n = nums.Length;
    int[] ans =new int[n];
    int s = 0;
    foreach (int x in nums) s ^= x;    
    for (int i = n - 1; i >= 0; --i) {
      ans[n - i - 1] = t ^ s;
      s ^= nums[i];
    }
    return ans;
    }
}

// 1828. Queries on Number of Points Inside a Circle
/*Solution: Brute Force
Time complexity: O(P * Q)
Space complexity: O(1)*/
public class Solution {
    public int[] CountPoints(int[][] points, int[][] queries) {
        List<int> ans = new List<int>(queries.Length);
    //ans.reserve(queries.size());
    foreach (int[] q in queries) {
       int rs = q[2] * q[2];
      int cnt = 0;      
      foreach (int[] p in points)
        if ((q[0] - p[0]) * (q[0] - p[0]) + 
            (q[1] - p[1]) * (q[1] - p[1]) <= rs)
          ++cnt;
      ans.Add(cnt);
    }
    return ans.ToArray();
    }
}

// 1827. Minimum Operations to Make the Array Increasing
/*Solution: Track Last
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MinOperations(int[] nums) {
        int ans = 0, last = 0;
    foreach (int x in nums)    
      if (x <= last)
        ans += ++last - x;
      else
        last = x;      
    return ans;
    }
}

// 1825. Finding MK Average
/*Solution 1: Multiset * 3
Use three multiset to track the left part (smallest k elements), 
right part (largest k elements) and mid (middle part of m – 2*k elements).

Time complexity: addElememt: O(logn), average: O(1)
Space complexity: O(n)

*/
public class MKAverage {

    public MKAverage(int m1, int k1) {
        sum = 0; m = m1; k = k1; n = m1 - 2*k1;
    }
    
    public void AddElement(int num) {
        if (q.Count == m) {      
      remove(q.Dequeue());
      
    }
    q.Enqueue(num);
    add(num);
    }
    
    public int CalculateMKAverage() {
         return (q.Count < m) ? -1 : (int)sum / n;
    }
    private void add(int x) { 
        Insert(left, x);
    
    if (left.Count > k) {
      int it = left[left.Count - 1];
      sum += it;
        Insert(mid, it);     
      left.Remove(it);
    }
    
    if (mid.Count > n) {
      int it = mid[mid.Count - 1];
      sum -= it; 
        Insert(right, it);
      mid.Remove(it);
    }
  }
  
  void remove(int x) {
    if (x <= left[left.Count -1]) {
        if (left.Contains(x))  left.Remove(x);
    } else if (x <= mid[mid.Count -1]) {
         if (mid.Contains(x)){
             sum -= x;
             mid.Remove(x);
         }
    } else {
         if (right.Contains(x))  right.Remove(x);
    }
    
    if (left.Count < k) {
      int it = mid[0];
      sum -= it; 
        Insert(left, it);
      mid.Remove(it);
    }
    
    if (mid.Count < n) {
      int it = right[0];
      sum += it; 
        Insert(mid, it);
      right.Remove(it);
    }
  }
    
    public void Insert(List<int> list, int item) {  
        
        int index = list.BinarySearch(item);
        
        if (index < 0) index = ~index;
        
        list.Insert(index, item);
    }
    
  public Queue<int> q = new Queue<int>();
  public List<int> left = new List<int>(), mid = new List<int>(), right = new List<int>();  
  public long sum = 0;
  public int m = 0;
  public int k = 0;
  public int n = 0;
}

/**
 * Your MKAverage object will be instantiated and called as such:
 * MKAverage obj = new MKAverage(m, k);
 * obj.AddElement(num);
 * int param_2 = obj.CalculateMKAverage();
 */

// 1824. Minimum Sideway Jumps
/*Solution: DP
Time complexity: O(n*k)
Space complexity: O(n*k) -> O(k)*/
public class Solution {
    public int MinSideJumps(int[] obstacles) {
        int n = obstacles.Length;
    int[] dp = new int[3]{1, 0, 1};    
    foreach (int o in obstacles) {
      if (o > 0) dp[o - 1] = (int)1e9;
      for (int k = 0; k < 3; ++k) {      
        if (k == o - 1) continue;
        dp[k] = Math.Min(Math.Min(dp[k], 
                     dp[(k + 1) % 3] + 1), 
                     dp[(k + 2) % 3] + 1);
      }
    }
    return dp.Min();
    }
}

// 1819. Number of Different Subsequences GCDs
/*Solution: Math
Enumerate all possible gcds (1 to max(nums)), 
and check whether there is a subset of the numbers 
that can form a given gcd i.
If we want to check whether 10 is a valid gcd, 
we found all multipliers of 10 in the array and compute their gcd.
ex1 gcd(10, 20, 30) = 10, true
ex2 gcd(20, 40, 80) = 20, false

Time complexity: O(mlogm)
Space complexity: O(m)*/
public class Solution {
    public int CountDifferentSubsequenceGCDs(int[] nums) {
    int kMax = nums.Max();
    int[] s = new int[kMax + 1];
    foreach (int x in nums) s[x] = 1;    
    int ans = 0;
    for (int i = 1; i <= kMax; ++i) {
      int g = 0;
      for (int j = i; j <= kMax; j += i)
        if (s[j] > 0) g = gcd(g, j);
      ans += Convert.ToInt32(g == i);
    }
    return ans;
    }
    private int gcd(int a, int b) {
        return b > 0 ? gcd(b, a % b) : a;
    }
}

// 1817. Finding the Users Active Minutes
/*Solution: Hashsets in a Hashtable
key: user_id, value: set{time}

Time complexity: O(n + k)
Space complexity: O(n + k)*/
public class Solution {
    public int[] FindingUsersActiveMinutes(int[][] logs, int k) {
        Dictionary<int, HashSet<int>> m = new Dictionary<int, HashSet<int>>();
    int[] ans = new int[k];
    foreach (int[] log in logs){
        m[log[0]] = m.GetValueOrDefault(log[0], new HashSet<int>()); m[log[0]].Add(log[1]);
    }
      
    foreach (var (id, s) in m)
      ++ans[s.Count - 1];
    return ans;
    }
}

// 1816. Truncate Sentence
/*Solution:
Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public string TruncateSentence(string s, int k) {
        StringBuilder ans = new StringBuilder();;
    String[] ss = s.Split(" ");
    for (int i = 0; i < k && ss.Length > 0; ++i) {

      ans.Append((ans.Length == 0 ? "" : " ") + ss[i]);
    }
    return ans.ToString()
       
    }
}

// 1815. Maximum Number of Groups Getting Fresh Donuts
/*Solution 1: Recursion w/ Memoization

State: count of group size % batchSize*/
public class Solution {
    public int MaxHappyGroups(int batchSize, int[] groups) {
        int ans = 0;
    int[] count = new int[batchSize];
    foreach (int g in groups) {
      int r = g % batchSize;
      if (r == 0) { ++ans; continue; }      
      if (count[batchSize - r] != 0) {
        --count[batchSize - r];
        ++ans;
      } else {
        ++count[r];
      }
    }    
    Dictionary<String, int> cache = new Dictionary<String, int>();

    int dp(int r, int n) {
      if (n == 0) return 0;
    string countStr = String.Join(',', count);
    if (cache.ContainsKey(countStr) )
      return cache[countStr];
    
      int ans = 0;
      for (int i = 1; i < batchSize; ++i) {
        if (count[i] == 0) continue;
        --count[i];
        ans = Math.Max(ans, Convert.ToInt32(r == 0) + dp((r +  batchSize - i) %  batchSize, n - 1));
        ++count[i];
      }
      return cache[countStr] = ans;
    };
        
   int n = count.Sum();//0; Array.ForEach(count, i => n += i);// <= Slow//accumulate(begin(count), end(count), 0);
        
    return ans + ((n > 0) ? dp(0, n) : 0);
    }
}

// SortedDictionary<int[], int> => when int[] is too long
// this would give TLE
// use Dictionary<String, int> is much better , string is good
public class Solution {
    public int MaxHappyGroups(int batchSize, int[] groups) {
        int ans = 0;
    int[] count = new int[batchSize];
    foreach (int g in groups) {
      int r = g % batchSize;
      if (r == 0) { ++ans; continue; }      
      if (count[batchSize - r] != 0) {
        --count[batchSize - r];
        ++ans;
      } else {
        ++count[r];
      }
    }    
    SortedDictionary<int[], int> cache = new SortedDictionary<int[], int>();

    int dp(int r, int n) {
      if (n == 0) return 0;
    
    if (cache.ContainsKey(count) && cache[count] != cache.Last().Value)
      return cache[count];
    
      int ans = 0;
      for (int i = 1; i < batchSize; ++i) {
        if (count[i] == 0) continue;
        --count[i];
        ans = Math.Max(ans, Convert.ToInt32(r == 0) + dp((r +  batchSize - i) %  batchSize, n - 1));
        ++count[i];
      }
      return cache[count] = ans;
    };
        
   int n = count.Sum();//0; Array.ForEach(count, i => n += i);//accumulate(begin(count), end(count), 0);
        
    return ans + ((n > 0) ? dp(0, n) : 0);
    }
}
// 1814. Count Nice Pairs in an Array
/*Solution: Two Sum
Key = x – rev(x)

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int CountNicePairs(int[] nums) {
        int kMod = (int)1e9 + 7;
    Dictionary<int, int> m = new  Dictionary<int, int>();
    long ans = 0;
    foreach (int x in nums) {      
        char[] array = x.ToString().ToCharArray() ;
Array.Reverse(array) ;
 String s = new string(array) ;
         m[x - Convert.ToInt32(s)] = m.GetValueOrDefault(x - Convert.ToInt32(s), 0);
      ans += m[x - Convert.ToInt32(s)]++;
    }
    return Convert.ToInt32(ans % kMod);
    }
}

//Linq
public class Solution {
    public int CountNicePairs(int[] nums) => (int)nums
  .GroupBy(x => x - x.ToString().Reverse().Aggregate(0, (a, b) => 10 * a + b - '0'))
  .Select(g => (long)g.Count())
  .Aggregate(0L, (a, b) => (a + b * (b - 1) / 2) % 1_000_000_007);
}

// 1813. Sentence Similarity III
/*Solution: Dequeue / Common Prefix + Suffix
Break sequences to words, store them in two deques. 
Pop the common prefix and suffix. 
At least one of the deque should be empty.

Time complexity: O(m+n)
Space complexity: O(m+n)

*/
public class Solution {
    public bool AreSentencesSimilar(string sentence1, string sentence2) {

    List<string> w1 = sentence1.Split(" ").Reverse().ToList(), w2 = sentence2.Split(" ").Reverse().ToList();
    while (w1.Count > 0 && w2.Count > 0 && w1.First() == w2.First()){
        w1.RemoveAt(0); w2.RemoveAt(0);
    }
      
    while (w1.Count > 0 && w2.Count > 0 && w1.Last() == w2.Last()){
        w1.RemoveAt(w1.Count - 1); w2.RemoveAt(w2.Count - 1);
    }
      
    return w1.Count == 0 || w2.Count == 0;
    
    }
}
//LinkedList <= C# Deque repacement
public class Solution {
    public bool AreSentencesSimilar(string sentence1, string sentence2) {
       LinkedList<string> dq1 = new LinkedList<string>(sentence1.Split(" "));
        LinkedList<string> dq2 = new LinkedList<string>(sentence2.Split(" "));

        while (dq1.Count != 0 && dq2.Count != 0 && dq1.First.Value.Equals(dq2.First.Value))
        {
            dq1.RemoveFirst();
            dq2.RemoveFirst();
        }

        while (dq1.Count != 0 && dq2.Count != 0 && dq1.Last.Value.Equals(dq2.Last.Value))
        {
            dq1.RemoveLast();
            dq2.RemoveLast();
        }
        return dq1.Count == 0 || dq2.Count == 0;
    }
}
// 1812. Determine Color of a Chessboard Square
/*Solution: Mod2
return (row_index + col_index) % 2 == 0

Time complexity: O(1)
Space complexity: O(1)*/
public class Solution {
    public bool SquareIsWhite(string coordinates) {
        return ((coordinates[0] - 'a') + (coordinates[1] - '0')) % 2 == 0;
    }
}

// 1808. Maximize Number of Nice Divisors
/*Solution: Math
Time complexity: O(logn)
Space complexity: O(1)*/
public class Solution {
    public int MaxNiceDivisors(int primeFactors) {
        int kMod = (int)1e9 + 7;
    long powm(long b, int exp) {
      long ans = 1;
      while (exp > 0) {
        if ((exp & 1) > 0) ans = (ans * b) % kMod;
        b = (b * b) % kMod;
        exp >>= 1;
      }
      return ans;
    };
    
    if (primeFactors <= 3) return primeFactors;
    switch (primeFactors % 3) {
      case 0: return Convert.ToInt32(powm(3, primeFactors / 3));
      case 1: return Convert.ToInt32((powm(3, primeFactors / 3 - 1) * 4) % kMod);
      case 2: return Convert.ToInt32((powm(3, primeFactors / 3) * 2) % kMod);
    }
    return -1;
    }
}

// 1807. Evaluate the Bracket Pairs of a String
/*Solution: Hashtable + Simulation
Time complexity: O(n+k)
Space complexity: O(n+k)

*/
public class Solution {
    public string Evaluate(string s, IList<IList<string>> knowledge) {
        Dictionary<string, string> m = new Dictionary<string, string>();
    foreach (var p in knowledge)
      m[p[0]] = m.GetValueOrDefault(p[0], p[1]);
    StringBuilder ans = new StringBuilder();
    StringBuilder cur = new StringBuilder();
    bool i = false;
    foreach (char c in s) {
      if (c == '(') {
        i = true;       
      } else if (c == ')') {
        if (m.ContainsKey(cur.ToString()))
          ans.Append(m[cur.ToString()]);
        else
          ans.Append("?");
        cur.Clear();
        i = false;      
      } else {
        if (!i) ans.Append(c);
        else cur.Append(c);
      }
    }
    return ans.ToString();
    }
}

// 1806. Minimum Number of Operations to Reinitialize a Permutation
/*Solution: Brute Force / Simulation
Time complexity: O(n2) ?
Space complexity: O(n)

*/
public class Solution {
    public int ReinitializePermutation(int n) {
         int[] perm = new int[n];
    int[] arr = new int[n];
    for (int i = 0; i < n; ++i) perm[i] = i;
    int ans = 0;
    bool flag = true;
    while (flag && ++ans > 0) {
      flag = false;
      for (int i = 0; i < n; ++i) {
        arr[i] = ((i & 1) > 0) ? perm[n / 2 + (i - 1) / 2] : perm[i / 2];
        flag |= arr[i] != i;        
      }
      //swap(perm, arr);
        int[] temp = perm;
        perm = arr;
        arr = temp;
    }
    return ans;
    }
}

// 1805. Number of Different Integers in a String
/*Solution: Hashtable
Be careful about leading zeros.

Time complexity: O(n)
Space complexity: O(n)

*/
// use HashSet + List
public class Solution {
    public int NumDifferentIntegers(string word) {
        word += "$";
    HashSet<string> s = new HashSet<string>();
    List<char> cur = new List<char>();
    foreach (char c in word) {
      if (char.IsNumber(c)) {       
        cur.Add(c);
      } else if (cur.Count != 0) {      
        while (cur.Count > 1 && cur[0] == '0')
          cur.RemoveAt(0);
        s.Add(String.Join(", ", cur));
        cur.Clear();
      }
    }
        //Console.WriteLine("s: "+ String.Join(", ", s.ToList()));
    return s.Count;
    }
}
// use HashSet + String
public class Solution {
    public int NumDifferentIntegers(string word) {
       HashSet<string> hs = new HashSet<string>();
        String buffer = "";
        
        for (int i = 0; i < word.Length; i++){
            if(Char.IsDigit(word,i)){
                buffer += word[i];
            }
            else if (buffer.Length > 0){
                hs.Add(buffer.TrimStart('0'));
                buffer = "";
            }
        }
        
        if (buffer.Length > 0){
            hs.Add(buffer.TrimStart('0'));
        }
        
        return hs.Count;
    }
}
// use HashSet + StringBuilder
public class Solution {
    public int NumDifferentIntegers(string word) {
       HashSet<string> set = new HashSet<string>();
        
        int i = 0;     
        while(i < word.Length)
        {
            if(word[i] >= 'a' && word[i] <= 'z')
            {
                i++;
                continue;
            }
            
            StringBuilder sb = new StringBuilder();
            while(i < word.Length && word[i] >= '0' && word[i] <= '9')
                sb.Append(word[i++]);

            set.Add(sb.ToString().TrimStart('0')); // "0", "00", "000" will be trimed to empty string
            
        }
        
        return set.Count;
    }
}
// 1801. Number of Orders in the Backlog
/* 
Solution: Treemap / PriorityQueue / Heap
buy backlog: max heap
sell backlog: min heap
Trade happens between the tops of two queues.

Time complexity: O(nlogn)
Space complexity: O(n)

Priority Queue

Complexity
Time O(nlogn)
Space O(n)*/
public class Solution {
    public int GetNumberOfBacklogOrders(int[][] orders) {
       PriorityQueue<int[],int[]> buy = new PriorityQueue<int[],int[]>(Comparer<int[]>.Create((a, b) => b[0] - a[0]));
        PriorityQueue<int[],int[]> sell = new PriorityQueue<int[],int[]>(Comparer<int[]>.Create((a, b) => a[0] - b[0]));
        foreach (int[] o in orders) {
            if (o[2] == 0)
                buy.Enqueue(o, o);
            else
                sell.Enqueue(o, o);
            while (buy.Count != 0 && sell.Count != 0 && sell.Peek()[0] <= buy.Peek()[0]) {
                int k = Math.Min(buy.Peek()[1], sell.Peek()[1]);
              //  buy.Peek()[1] -= k;
              //  sell.Peek()[1] -= k;
                if ((buy.Peek()[1] -=k) == 0) buy.Dequeue();
                if ((sell.Peek()[1] -=k) == 0) sell.Dequeue();
            }

        }
        int res = 0, mod = 1000000007;
        while (sell.Count > 0)
            res = (res + sell.Dequeue()[1]) % mod;
       while (buy.Count > 0)
            res = (res + buy.Dequeue()[1]) % mod;
        return res;
    }
}

//Dictionary would result in TLE
public class Solution {
    public int GetNumberOfBacklogOrders(int[][] orders) {
        int kMod = (int)1e9 + 7;
     SortedDictionary<Int64, Int64> buys = new SortedDictionary<Int64, Int64>();
    SortedDictionary<Int64, Int64> sells = new SortedDictionary<Int64, Int64>();
    foreach (int[] order in orders) {      
      if(order[2] == 0 ) buys[order[0]] = buys.GetValueOrDefault(order[0], 0) + order[1]; else sells[order[0]] = sells.GetValueOrDefault(order[0], 0) + order[1];
      //m[order[0]] += order[1];
       //buys = buys.OrderByDescending(u => u.Key).ToDictionary(z => z.Key, y => y.Value);
       // sells = sells.OrderBy(u => u.Key).ToDictionary(z => z.Key, y => y.Value);
      while (buys.Count > 0 && sells.Count > 0) {
        var b = buys.Last().Key;//ElementAt(buys.Count - 1);
        var s = sells.First().Key;//ElementAt(0);
        if (b < s) break;
        Int64 k = Math.Min(buys[b], sells[s]);
        //buys[b] -= k; sells[s] -= k;
        if ((buys[b] -= k) == 0) buys.Remove(++b); // (++b).base()
        if ((sells[s] -= k) == 0) sells.Remove(s);
      }
    }
    Int64 ans = 0;
    foreach (var (p, c) in buys) ans += c;
    foreach (var (p, c) in sells) ans += c;    
    return (int)(ans % kMod);
    }
}
// 1800. Maximum Ascending Subarray Sum
/*Solution: Running sum with resetting
Time complexity: O(n)
Space complexity: O(1)

Track the running sum and reset it to zero if nums[i] <= nums[i – 1]*/
public class Solution {
    public int MaxAscendingSum(int[] nums) {
        int ans = 0;
        for (int i = 0, s = 0; i < nums.Length; ++i) {
        if (i > 0 && nums[i] <= nums[i - 1])
            s = 0;
        ans = Math.Max(ans, s += nums[i]);
        }
        return ans;
    }
}

// 1799. Maximize Score After N Operations
/*Solution: Mask DP

dp(mask, i) := max score of numbers (represented by a binary mask) at the i-th operations.
ans = dp(1, mask)
base case: dp = 0 if mask == 0
Transition: dp(mask, i) = max(dp(new_mask, i + 1) + i * gcd(nums[m], nums[n]))

Time complexity: O(n2*22n)
Space complexity: O(22n)

Bottom-Up
*/
using System.Numerics;
public class Solution {
    public int MaxScore(int[] nums) {
        int l = nums.Length;
    int[] dp = new int[1 << l];    
    for (int mask = 0; mask < (1 << l); ++mask) {
      int c = BitOperations.PopCount((uint)mask);// __builtin_popcount(mask);
      if ((c & 1) > 0 ) continue; // only do in pairs
      int k = c / 2 + 1;
      for (int i = 0; i < l; ++i)
        for (int j = i + 1; j < l; ++j)
          if ((mask & (1 << i)) + (mask & (1 << j)) == 0) {            
            int new_mask = mask | (1 << i) | (1 << j);            
            dp[new_mask] = Math.Max(dp[new_mask],
                               k * gcd(nums[i], nums[j]) + dp[mask]);
        }
    }
    return dp[(1 << l) - 1];
    }
     private int gcd(int a, int b) {
        return b > 0 ? gcd(b, a % b) : a;
    }
}

// 1798. Maximum Number of Consecutive Values You Can Make
/*Solution: Greedy + Math
We want to start with smaller values, sort input array in ascending order.

First of all, the first number has to be 1 in order to generate sum of 1.
Assuming the first i numbers can generate 0 ~ k.
Then the i+1-th number x can be used if and only if x <= k + 1, 
such that we can have a consecutive sum of k + 1 by adding x to 
a sum between [0, k] and the new maximum sum we have will be k + x.

Time complexity: O(nlogn)
Space complexity: O(1)*/
public class Solution {
    public int GetMaximumConsecutive(int[] coins) {
    Array.Sort(coins, (a,b) => a - b);
    int ans = 0;
    foreach (int c in coins) {      
      if (c > ans + 1) break;
      ans += c;
    }
    return ans + 1;
    }
}

// 1796. Second Largest Digit in a String
/*Solution: Hashtable
Use a hashtable to store the token and its expiration time.

Time complexity: at most O(n) per operation
Space complexity: O(n)

*/
public class Solution {
    public int SecondHighest(string s) {
         int[] d = new int[10];
    foreach (char c in s)
      if (c >= '0' && c <= '9')
        d[c - '0'] = 1;
    int order = 0;
    for (int i = 9; i >= 0; --i)
      if (d[i] > 0 && ++order == 2)
        return i;
    return -1;
    }
}

// 1793. Maximum Score of a Good Subarray
/*Solutions: Two Pointers
maintain a window [i, j], m = min(nums[i~j]), expend to the left if nums[i – 1] >= nums[j + 1], otherwise expend to the right.

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int MaximumScore(int[] nums, int k) {
        int n = nums.Length;    
    int ans = 0;
    for (int i = k, j = k, m = nums[k];;) {
      ans = Math.Max(ans, m * (j - i + 1));      
      if (j - i + 1 == n) break;
      int l = (i > 0) ? nums[i - 1] : -1;
      int r = (j + 1 < n) ? nums[j + 1] : -1;
      if (l >= r)
        m = Math.Min(m, nums[--i]);
      else
        m = Math.Min(m, nums[++j]);      
    }
    return ans;
    }
}

// 1792. Maximum Average Pass Ratio
/*Solution: Greedy + Heap

Sort by the ratio increase potential (p + 1) / (t + 1) – p / t.

Time complexity: O((m+n)logn)
Space complexity: O(n)*/
// use KeyValuePair
// use CompareTo to compare Double
public class Solution {
    public double MaxAverageRatio(int[][] classes, int extraStudents) {
        int n = classes.Length;
    double ratio(int i, int delta = 0) {
      return (double)(classes[i][0] + delta) / 
        (double)(classes[i][1] + delta);
    };
    PriorityQueue<KeyValuePair<double, int>, KeyValuePair<double, int>> q = new  PriorityQueue<KeyValuePair<double, int>, KeyValuePair<double, int>>(Comparer<KeyValuePair<double,int>>.Create((a, b) => b.Key.CompareTo(a.Key)));
    for (int i = 0; i < n; ++i)
      q.Enqueue(new KeyValuePair<double,int>(ratio(i, 1) - ratio(i), i), new KeyValuePair<double,int>(ratio(i, 1) - ratio(i), i));
    while (extraStudents-- > 0) {
     var (r, i) = q.Peek();q.Dequeue();      
      ++classes[i][0];
      ++classes[i][1];
      q.Enqueue(new KeyValuePair<double,int>(ratio(i, 1) - ratio(i), i),new KeyValuePair<double,int>(ratio(i, 1) - ratio(i), i) );
    }
    double total_ratio = 0;
    for (int i = 0; i < n; ++i)
      total_ratio += ratio(i);
    return total_ratio / n;
    }
}
// use Tuple 
// use CompareTo to compare Double
public class Solution {
    public double MaxAverageRatio(int[][] classes, int extraStudents) {
        int n = classes.Length;
    double ratio(int i, int delta = 0) {
      return (double)(classes[i][0] + delta) / 
        (double)(classes[i][1] + delta);
    };
    PriorityQueue<(double r, int i), (double r, int i)> q = new PriorityQueue<(double r, int i), (double r, int i)>(Comparer<(double r, int i)>.Create((a, b) => b.r.CompareTo(a.r)));
    for (int i = 0; i < n; ++i)
      q.Enqueue((ratio(i, 1) - ratio(i), i), (ratio(i, 1) - ratio(i), i));
    while (extraStudents-- > 0) {
     var (r, i) = q.Peek();q.Dequeue();      
      ++classes[i][0];
      ++classes[i][1];
      q.Enqueue((ratio(i, 1) - ratio(i), i),(ratio(i, 1) - ratio(i), i) );
    }
    double total_ratio = 0;
    for (int i = 0; i < n; ++i)
      total_ratio += ratio(i);
    return total_ratio / n;
    }
}

// 1791. Find Center of Star Graph
/*Solution: Graph / Hashtable
Count the degree of each node, return the one with n-1 degrees.

Time complexity: O(n)
Space complexity: O(n)

*/
public class Solution {
    public int FindCenter(int[][] edges) {
        Dictionary<int, int> degrees = new ();
    foreach (var e in edges) {
      degrees[e[0]] = degrees.GetValueOrDefault(e[0], 0) + 1;
      degrees[e[1]] = degrees.GetValueOrDefault(e[1], 0) + 1;
    }
    int n = degrees.Count;
    foreach (var (id, d) in degrees)
      if (d == n - 1) return id;
    return -1;
    }
}

// 1786. Number of Restricted Paths From First to Last Node
/*Solution: Dijkstra + DFS w/ memoization

Find shortest path from n to all the nodes.
paths(u) = sum(paths(v)) if dist[u] > dist[v] and (u, v) has an edge
return paths(1)

Time complexity: O(ElogV + V + E)
Space complexity: O(V + E)

*/
// tuple (int d, int u)
public class Solution {
    public int CountRestrictedPaths(int n, int[][] edges) {
         int kMod = (int)1e9 + 7;
    //using PII = KeyValuePair<int, int>;    
    List<(int d, int u)>[] g = new List<(int d, int u)>[n + 1];
        //Array.Fill(g, new List<(int d, int u)>()); <= WON'T WORK WILL CAUSE ERROR 
        for(int i = 0; i < n+1; i++)
            g[i] = new List<(int d, int u)>();
    foreach (var e in edges) {
      g[e[0]].Add((e[1], e[2]));
      g[e[1]].Add((e[0], e[2]));
    }    
    
    // Shortest distance from n to x.
    int[] dist = new int[n + 1] ; Array.Fill(dist, Int32.MaxValue);
    int[] dp = new int[n + 1]; //Array.Fill(dp, 0);
    dist[n] = 0;
    dp[n] = 1;
    PriorityQueue<(int d, int u), (int d, int u)> q = new PriorityQueue<(int d, int u), (int d, int u)>(Comparer<(int d, int u)>.Create((a, b) => a.d - b.d ));
    
    q.Enqueue((0, n), (0, n));
    while (q.Count > 0) {
      var (d, u) = q.Peek(); q.Dequeue();
      if (d > dist[u]) continue;
     if (u == 1) break; //prunning
      foreach (var (v, w) in g[u]) {
        if (dist[v] > dist[u] + w){
            dist[v] = dist[u] + w;
            q.Enqueue((dist[v] , v),(dist[v] , v));
        }
          
        if (dist[v] > dist[u])
          dp[v] = (dp[v] + dp[u]) % kMod;
      }
    }    
    return dp[1];
    }
}
// KeyValuePair
public class Solution {
    public int CountRestrictedPaths(int n, int[][] edges) {
         int kMod = (int)1e9 + 7;
    //using PII = KeyValuePair<int, int>;    
    List<KeyValuePair<int, int>>[] g = new List<KeyValuePair<int, int>>[n + 1];
       // Array.Fill(g, new List<KeyValuePair<int, int>>()); <= WON'T WORK WILL CAUSE ERROR 
         for(int i = 0; i < n+1; i++)
            g[i] = new List<KeyValuePair<int, int>>();
    foreach (var e in edges) {
      g[e[0]].Add(new KeyValuePair<int, int>(e[1], e[2]));
      g[e[1]].Add(new KeyValuePair<int, int>(e[0], e[2]));
    }    
    
    // Shortest distance from n to x.
    int[] dist = new int[n + 1] ; Array.Fill( dist, Int32.MaxValue);
    int[] dp = new int[n + 1]; 
    dist[n] = 0;
    dp[n] = 1;
    PriorityQueue<KeyValuePair<int, int>, KeyValuePair<int, int>> q = new PriorityQueue<KeyValuePair<int, int>, KeyValuePair<int, int>>(Comparer<KeyValuePair<int,int>>.Create((a, b) => a.Key - b.Key));//b.Key == a.Key ? b.Value - a.Value : b.Key - a.Key));
    
    q.Enqueue(new KeyValuePair<int, int>(0, n), new KeyValuePair<int, int>(0, n));
    while (q.Count != 0) {
      var (d, u) = q.Peek(); q.Dequeue();
      if (d > dist[u]) continue;
      if (u == 1) break;
      foreach (var (v, w) in g[u]) {
        if (dist[v] > dist[u] + w){
            dist[v] = dist[u] + w;
            q.Enqueue(new KeyValuePair<int, int>(dist[v] , v), new KeyValuePair<int, int>(dist[v] , v));
        }
          
        if (dist[v] > dist[u])
          dp[v] = (dp[v] + dp[u]) % kMod;
      }
    }    
    return dp[1];
    }
}
// SortedSet<(int d, int u)>
public class Solution {
    public int CountRestrictedPaths(int n, int[][] edges) {
          int kMod = (int)1e9 + 7;
    //using PII = KeyValuePair<int, int>;    
    List<(int d, int u)>[] g = new List<(int d, int u)>[n + 1];
        //Array.Fill(g, new List<(int d, int u)>());
        for(int i = 0; i < n+1; i++)
            g[i] = new List<(int d, int u)>();
    foreach (var e in edges) {
      g[e[0]].Add((e[1], e[2]));
      g[e[1]].Add((e[0], e[2]));
    }    
    
    // Shortest distance from n to x.
    int[] dist = new int[n + 1] ; Array.Fill(dist, Int32.MaxValue);
    int[] dp = new int[n + 1]; //Array.Fill(dp, 0);
    dist[n] = 0;
    dp[n] = 1;
    SortedSet<(int d, int u)> q = new SortedSet<(int d, int u)>();
    // DON'T NEED TO PUT SortedSet Comparer<(int d, int u)>.Create((a, b) => a.d - b.d )
    // AT HERE, WILL CAUSE ERROR
    
    q.Add((0, n));
    while (q.Count > 0) {
      var (d, u) = q.First(); q.Remove(q.First());
      if (d > dist[u]) continue;
     if (u == 1) break; //prunning
      foreach (var (v, w) in g[u]) {
        if (dist[v] > dist[u] + w){
            dist[v] = dist[u] + w;
            q.Add((dist[v] , v));
        }
          
        if (dist[v] > dist[u])
          dp[v] = (dp[v] + dp[u]) % kMod;
      }
    }    
    return dp[1];
    }
}

// 1785. Minimum Elements to Add to Form a Given Sum
/*Solution: Math
Time complexity: O(n)
Space complexity: O(1)

Compute the diff = abs(sum(nums) – goal)
ans = （diff + limit – 1)) / limit*/
public class Solution {
    public int MinElements(int[] nums, int limit, int goal) {
        // accumulate(begin(nums), end(nums), 0LL)
        long sum = 0;  Array.ForEach(nums, i => sum += i);
        Int64 diff = Math.Abs(goal - sum);
    return (int)((diff + limit - 1) / limit);
    }
}

// 1784. Check if Binary String Has at Most One Segment of Ones
/*Solution: Counting
increase counter if s[i] == ‘1’ otherwise, reset counter.
increase counts when counter becomes 1.
return counts == 1

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public bool CheckOnesSegment(string s) {
        int count = 0;
    int ones = 0;
    foreach (char c in s) {
      if (c == '1') {
        count += Convert.ToInt32(++ones == 1);        
      } else {
        ones = 0;
      }
    }
    return count == 1;
    }
}

// 1782. Count Pairs Of Nodes
/*Solution 1: Pre-compute
Pre-compute # of pairs with total edges >= k. 
where k is from 0 to max_degree * 2 + 1.

Time complexity: (|node_degrees|2 + V + E)
Space complexity: O(V+E)

*/
public class Solution {
    public int[] CountPairs(int n, int[][] edges, int[] queries) {
        int[] node_degrees = new int[n];
    Dictionary<int, int> edge_freq = new  Dictionary<int, int>();
    foreach (var e in edges) {
      Array.Sort(e);
      ++node_degrees[--e[0]];
      ++node_degrees[--e[1]];
      edge_freq[(e[0] << 16) | e[1]] = edge_freq.GetValueOrDefault((e[0] << 16) | e[1] , 0) + 1;
    }
    
    int max_degree = node_degrees.Max();
    // Need pad one more to handle "not found / 0" case.
    int[] counts = new int[max_degree * 2 + 2];
    
     Dictionary<int, int> degree_count = new  Dictionary<int, int>();
    for (int i = 0; i < n; ++i) 
      degree_count[node_degrees[i]] =  degree_count.GetValueOrDefault(node_degrees[i], 0) + 1;
    
    foreach (var (d1, c1) in degree_count)
      foreach (var (d2, c2) in degree_count)
        // Only count once if degrees are different to ensure (a < b)
        if (d1 < d2) counts[d1 + d2] += c1 * c2;
        // If degrees are the same C(n, 2) to ensure (a < b)
        else if (d1 == d2) counts[d1 * 2] += c1 * (c1 - 1) / 2;
    
    foreach (var (key, freq) in edge_freq) {
      int u = key >> 16;
      int v = key & 0xFFFF;
      // For a pair of (u, v) their actual edge count is 
      // d[u] + d[v] - freq[(u, v)] instead of d[u] + d[v]
      counts[node_degrees[u] + node_degrees[v]] -= 1;
      counts[node_degrees[u] + node_degrees[v] - freq] += 1;
    }
    
    // counts[i] = # of pairs whose total edges >= i
    for (int i = counts.Length - 2; i >= 0; --i)
      counts[i] += counts[i + 1];
    
    List<int> ans = new List<int>();
    foreach (int q in queries)
      ans.Add(counts[Math.Min(q + 1, (int)(counts.Length - 1))]);
    return ans.ToArray();
    }
}

// 1781. Sum of Beauty of All Substrings
/*Solution: Treemap
Time complexity: O(n2log26)
Space complexity: O(26)*/
public class Solution {
    public int BeautySum(string s) {
        int n = s.Length;
    int ans = 0;
    int[] f = new int[26];
    SortedDictionary<int, int> m = new SortedDictionary<int, int>();
    for (int i = 0; i < n; ++i) {
        Array.Fill(f, 0);
      //fill(begin(f), end(f), 0);
      m.Clear();
      for (int j = i; j < n; ++j) {
        int c = ++f[s[j] - 'a'];
        m[c] = m.GetValueOrDefault(c,0) + 1;
        if (c > 1 && m.ContainsKey(c - 1)) {
          //int it = m[c - 1];
          if (--m[c - 1] == 0)
            m.Remove(c - 1);
        }
        ans += m.Last().Key - m.First().Key;  
      }
    }
    return ans;
    }
}

// 1780. Check if Number is a Sum of Powers of Three
/*Solution: Greedy + Math

Find the largest 3^x that <= n, subtract that from n and repeat the process.
x should be monotonically decreasing, otherwise we have duplicate terms.
e.g. 12 = 32 + 31 true
e.g. 21 = 32 + 32+ 31 false

Time complexity: O(logn?)
Space complexity: O(1)*/
public class Solution {
    public bool CheckPowersOfThree(int n) {
         int last = Int32.MaxValue;
    while (n > 0) {
      int p = 0;
      int cur = 1;
      while (cur * 3 <= n) {
        cur *= 3;
        ++p;
      }
      if (p == last) return false;
      last = p;    
      n -= cur;
    }
    return true;
    }
}

// 1773. Count Items Matching a Rule
/*Solution: Brute Force
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int CountMatches(IList<IList<string>> items, string ruleKey, string ruleValue) {
      int res = 0;
        
        foreach(var item in items){
            if(ruleKey == "type" && item[0] == ruleValue
        || ruleKey == "color" && item[1] == ruleValue
        || ruleKey == "name" && item[2] == ruleValue) res++;
        }
        
        return res;
    }
}

// 1771. Maximize Palindrome Length From Subsequences
/*Solution: DP

Similar to 花花酱 LeetCode 516. Longest Palindromic Subsequence

Let s = word1 + word2, build dp table on s. 
We just need to make sure there’s at least one char from each string.

Time complexity: O((m+n)^2)
O(m+n) Space complexity
*/
public class Solution {
    public int LongestPalindrome(string word1, string word2) {
        int l1 = word1.Length;
    int l2 = word2.Length;
    string s = word1 + word2;
    int n = l1 + l2;
    int[] dp1 = new int[n]; Array.Fill(dp1, 1);  int[] dp2 = new int[n];
    int ans = 0;
    for (int l = 2; l <= n; ++l) {
       int[] dp = new int[n];
      for (int i = 0, j = i + l - 1; j < n; ++i, ++j) {
        if (s[i] == s[j]) {
          dp[i] = dp2[i + 1] + 2;
          if (i < l1 && j >= l1)
            ans = Math.Max(ans, dp[i]);
        } else {
          dp[i] = Math.Max(dp1[i + 1], dp1[i]);
        }
      }
      // dp2, dp1 = dp1, dp
     // dp1.swap(dp); 
     // dp2.swap(dp);
    int[] temp = dp1; dp1 = dp; dp = temp;
    temp = dp2; dp2 = dp; dp = temp;
    }
    return ans;
    }
}

// 1770. Maximum Score from Performing Multiplication Operations
/*Solution: DP

dp(i, j) := max score we can get with nums[i~j] left.

k = n – (j – i + 1)
dp(i, j) = max(dp(i + 1, j) + nums[i] * multipliers[k], dp(i, j-1) + nums[j] * multipliers[k])

Time complexity: O(m*m)
Space complexity: O(m*m)

Bottom-Up
*/
public class Solution {
    public int MaximumScore(int[] nums, int[] multipliers) {
        int m = multipliers.Length;
    int n = nums.Length;
    // dp[i][j] := max score of using first i elements and last j elements
    //vector<vector<int>> dp(m + 1, vector<int>(m + 1));
    int[][] dp = new int[m + 1][];  Array.Fill(dp,new int[m + 1]);
   // for (int i = 0; i < m+1; i++) dp[i] = new int[m + 1];
    for (int k = 1; k <= m; ++k)
      for (int i = 0, j = k - i; i <= k; ++i, --j)
        dp[i][j] = Math.Max((i > 0 ? dp[i - 1][j] + nums[i - 1] * multipliers[k - 1] : Int32.MinValue),
                       (j > 0 ? dp[i][j - 1] + nums[n - j] * multipliers[k - 1] : Int32.MinValue));
    int ans = Int32.MinValue;
    for (int i = 0; i <= m; ++i)
      ans = Math.Max(ans, dp[i][m - i]);
    return ans;
    }
}

// 1769. Minimum Number of Operations to Move All Balls to Each Box
/*Solution: Prefix Sum + DP
Time complexity: O(n)
Space complexity: O(n)

*/
public class Solution {
    public int[] MinOperations(string boxes) {
         int n = boxes.Length;
    int[] ans = new int[n];    
    for (int i = 0, c = 0, s = 0; i < n; ++i) {
      ans[i] += c;
      c += (s += boxes[i] - '0');
    }
    for (int i = n - 1, c = 0, s = 0; i >= 0; --i) {
      ans[i] += c;
      c += (s += boxes[i] - '0');
    }    
    return ans;
    }
}

// 1776. Car Fleet II
/*Solution: Monotonic Stack
Key observation: If my speed is slower than the speed of the previous car, not only mine but also all cars behind me will NEVER be able to catch/collide with the previous car. Such that we can throw it away.

Maintain a stack that stores the indices of cars with increasing speed.

Process car from right to left, for each car, pop the stack (throw the fastest car away) if any of the following conditions hold.
1) speed <= stack.top().speed
2) There are more than one car before me and it takes more than to collide the fastest car than time the fastest took to collide.

Time complexity: O(n)
Space complexity: O(n)

*/
public class Solution {
    public double[] GetCollisionTimes(int[][] cars) {
        double collide(int i, int j) {
      return (double)(cars[i][0] - cars[j][0]) /
        (cars[j][1] - cars[i][1]);
    };
    int n = cars.Length;
    double[] ans = new double[n]; Array.Fill(ans, -1);
    Stack<int> s = new Stack<int>();
    for (int i = n - 1; i >= 0; --i) {
      while (s.Count != 0 && (cars[i][1] <= cars[s.Peek()][1] ||
                           (s.Count > 1 && collide(i, s.Peek()) > ans[s.Peek()])))
        s.Pop();
      ans[i] = s.Count == 0 ? -1 : collide(i, s.Peek());
      s.Push(i);
    }
    return ans;
    }
}

// 1775. Equal Sum Arrays With Minimum Number of Operations
/*Solution: Greedy

Assuming sum(nums1) < sum(nums2),
sort both arrays
* scan nums1 from left to right, we need to increase the value form the smallest one.
* scan nums2 from right to left, we need to decrease the value from the largest one.
Each time, select the one with the largest delta to change.

e.g. nums1[i] = 2, nums[j] = 4, delta1 = 6 – 2 = 4, delta2 = 4 – 1 = 3, 
Increase 2 to 6 instead of decreasing 4 to 1.

Time complexity: O(mlogm + nlogn)
Space complexity: O(1)

*/
public class Solution {
    public int MinOperations(int[] nums1, int[] nums2) {
        int l1 = nums1.Length;
    int l2 = nums2.Length;
    if (Math.Min(l1, l2) * 6 < Math.Max(l1, l2)) return -1;
    int s1 = nums1.Sum();//accumulate(begin(nums1), end(nums1), 0);
    int s2 = nums2.Sum();//accumulate(begin(nums2), end(nums2), 0);
    if (s1 > s2) return MinOperations(nums2, nums1);
    Array.Sort(nums1);//sort(begin(nums1), end(nums1));
    Array.Sort(nums2);//sort(begin(nums2), end(nums2));
    int ans = 0;    
    int i = 0;
    int j = l2 - 1;
    while (s1 != s2) {      
      int diff = s2 - s1;      
      if (j == l2 || (i != l1 && 6 - nums1[i] >= nums2[j] - 1)) {
        int x = Math.Min(6, nums1[i] + diff);
        s1 += x - nums1[i++];        
      } else {
        int x = Math.Max(1, nums2[j] - diff);
        s2 += x - nums2[j--];      
      }
      ++ans;
    }
    return ans;
    }
}

// 1774. Closest Dessert Cost
/*Solution 2: DFS
Combination

Time complexity: O(3^m * n)
Space complexity: O(m)

*/
public class Solution {
    public int ClosestCost(int[] baseCosts, int[] toppingCosts, int target) {
         int m = toppingCosts.Length;
    int min_diff = Int32.MaxValue;
    int ans = Int32.MaxValue;
    void dfs(int s, int cur) {
      if (s == m) {
        foreach (int b in baseCosts) {
          int total = b + cur;
          int diff = Math.Abs(total - target);
          if (diff < min_diff 
              || diff == min_diff && total < ans) {
            min_diff = diff;
            ans = total;
          }
        }
        return;
      }      
      for (int i = s; i < m; ++i) {
        dfs(i + 1, cur);
        dfs(i + 1, cur + toppingCosts[i]);
        dfs(i + 1, cur + toppingCosts[i] * 2);
      }
    };    
    dfs(0, 0);
    return ans;
    }
}

// 1766. Tree of Coprimes
/*Solution: DFS + Stack
Pre-compute for coprimes for each number.

For each node, enumerate all it’s coprime numbers, 
find the deepest occurrence.

Time complexity: O(n * max(nums))
Space complexity: O(n)*/
public class Solution {
    public int[] GetCoprimes(int[] nums, int[][] edges) {
        int kMax = 50;
     int n = nums.Length;
    List<int>[] g = new List<int>[n]; //(n);
        for(int i = 0; i < n; i++)
            g[i] = new List<int>();
    foreach (var e in edges) {
      g[e[0]].Add(e[1]);
      g[e[1]].Add(e[0]);
    }    
   //  Console.WriteLine("here1");  
   List<int>[] coprime = new List<int>[kMax + 1]; //(kMax + 1);
         for(int i = 0; i < kMax + 1; i++)
            coprime[i] = new List<int>();
    for (int i = 1; i <= kMax; ++i)
      for (int j = 1; j <= kMax; ++j)
        if (gcd(i, j) == 1) coprime[i].Add(j);
   // Console.WriteLine("here2");  
    int[] ans = new int[n]; Array.Fill(ans, Int32.MaxValue);//(n, INT_MAX);
   // vector<vector<pair<int, int>>> p(kMax + 1);
       List<List<KeyValuePair<int, int>>> p = new List<List<KeyValuePair<int, int>>>();
         for(int i = 0; i < kMax + 1; i++)
            p.Add(new List<KeyValuePair<int, int>>());
       // Console.WriteLine("here3");  
    void dfs(int cur, int d) {    
      int max_d = -1;
      int ancestor = -1;
      foreach (int co in coprime[nums[cur]])
        if (p[co].Count != 0 && p[co].Last().Key > max_d) {
          max_d = p[co].Last().Key;
          ancestor = p[co].Last().Value;          
        }
      ans[cur] = ancestor;
      p[nums[cur]].Add(new KeyValuePair<int, int>(d, cur));
      foreach (int nxt in g[cur])
        if (ans[nxt] == Int32.MaxValue) dfs(nxt, d + 1);
      p[nums[cur]].RemoveAt(p[nums[cur]].Count - 1);
    };
    
    dfs(0, 0);
    return ans;
    }
    
    private int gcd(int a, int b) {
        return b > 0 ? gcd(b, a % b) : a;
    }
}

// 1765. Map of Highest Peak
/*Solution: BFS
h[y][x] = min distance of (x, y) to any water cell.

Time complexity: O(m*n)
Space complexity: O(m*n)*/
public class Solution {
    public int[][] HighestPeak(int[][] isWater) {
        int m = isWater.Length;
    int n = isWater[0].Length;
    int[][] ans = new int[m][];//(m, vector<int>(n, INT_MIN));
        for(int i = 0; i < m; i++){
            ans[i] = new int[n]; Array.Fill(ans[i], Int32.MinValue);
        }

    Queue<KeyValuePair<int, int>> q = new Queue<KeyValuePair<int, int>>();
    for (int y = 0; y < m; ++y)
      for (int x = 0; x < n; ++x)
        if (isWater[y][x] > 0) {
          q.Enqueue(new KeyValuePair<int, int>(x, y));
          ans[y][x] = 0;
        }
   int[] dirs = new int[5]{0, -1, 0, 1, 0};    
    while (q.Count != 0) {
      var (cx, cy) = q.Dequeue();// q.Peek(); q.Dequeue();
      for (int i = 0; i < 4; ++i) {
        int x = cx + dirs[i];
        int y = cy + dirs[i + 1];
        if (x < 0 || x >= n || y < 0 || y >= m) continue;
        if (ans[y][x] != Int32.MinValue) continue;
        ans[y][x] = ans[cy][cx] + 1;
        q.Enqueue(new KeyValuePair<int, int>(x, y));
      }      
    }
    return ans;
    }
}

// 1764. Form Array by Concatenating Subarrays of Another Array
/*Solution: Brute Force
Time complexity: O(n^2?)
Space complexity: O(1)

*/
public class Solution {
    public bool CanChoose(int[][] groups, int[] nums) {
        int n = nums.Length;
    int s = 0;
    foreach (var g in groups) {
      bool found = false;
      for (int i = s; i <= n - g.Length; ++i)
        if (search(g, nums, i)) {
          s = i + g.Length;
          found = true;
          break;
        }
      if (!found) return false;
    }
    return true;
    }
    private bool search(int[] group, int[] nums, int start) {
        for(int i=0;i<group.Length;i++) 
            if(group[i] != nums[i+start])
                return false;
        return true;
    }
}

// 1763. Longest Nice Substring
/*Solution: Brute Force Optimized 1:

Time complexity: O(n^2*26)
Space complexity: O(1)

*/
public class Solution {
    public string LongestNiceSubstring(string s) {
        int n = s.Length;    
    string ans = "";
    for (int i = 0; i < n; ++i) {
      int[] u = new int[26]; int[] l = new int[26];
      for (int j = i; j < n; ++j) {
        char c = s[j];
        if (char.IsUpper(c)) u[c - 'A'] = 1;
        else l[c - 'a'] = 1;
        if (Enumerable.SequenceEqual(u, l) && j - i + 1 > ans.Length)
          ans = s.Substring(i, j - i + 1);
      }
    }
    return ans;//new string(ans);
    }
}

// 1761. Minimum Degree of a Connected Trio in a Graph
/*Solution: Brute Force
Try all possible Trios.

Time complexity: O(n^3)
Space complexity: O(n^2)*/
public class Solution {
    public int MinTrioDegree(int n, int[][] edges) {
        BitArray[] g = new BitArray[n]; //Array.Fill(g, new BitArray(400)); CAN'T NOT USE THIS WON'T WORK!
        for(int i = 0; i < n; i++)
            g[i] = new BitArray(400, false);
        int[] cnt = new int[n];
    foreach (var e in edges) {
      g[e[0] - 1].Set(e[1] - 1, true);
      g[e[1] - 1].Set(e[0] - 1, true);
        ++cnt[e[0] - 1];
        ++cnt[e[1] - 1];
    }
    int ans = Int32.MaxValue;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        if (g[i][j])
          for (int k = j + 1; k < n; ++k)
            if (g[i][k] && g[j][k])
              ans = Math.Min(ans,  cnt[i] +  cnt[j] +  cnt[k] - 6);
    return ans == Int32.MaxValue ? -1 : ans;
    }
}

//HashSet works too but slow
public class Solution {
    public int MinTrioDegree(int n, int[][] edges) {
        HashSet<int>[] g = new HashSet<int>[n]; //Array.Fill(g, new HashSet<int>());
        for(int i = 0; i < n; i++)
            g[i] = new HashSet<int>();
    foreach (var e in edges) {
      g[e[0] - 1].Add(e[1] - 1);
      g[e[1] - 1].Add(e[0] - 1);
    }
    int ans = Int32.MaxValue;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j){
          if(g[i].Contains(j))
          for (int k = j + 1; k < n; ++k)
            if (g[i].Contains(k) && g[j].Contains(k))
              ans = Math.Min(ans,  g[i].Count +  g[j].Count +  g[k].Count - 6);
      }
        
    return ans == Int32.MaxValue ? -1 : ans;
    }
}

// 1727. Largest Submatrix With Rearrangements
/*Solution: DP + Sorting

Preprocess each column, for col j, matrix[i][j] := length consecutive ones of col j.

[0,0,1]    [0,0,1]
[1,1,1] => [1,1,2]
[1,0,1]    [2,0,3]
Then we enumerate ending row, for each ending row i, we sort row[i] in deceasing order

e.g. i = 2

[0,0,1]                  [-,-,-]
[1,1,2] sort by row 2 => [-,-,-]
[2,0,3]                  [3,2,0]
row[2][1] = 3, means there is a 3×1 all ones sub matrix, area = 3
row[2][2] = 2, means there is a 2×2 all ones sub matrix, area = 4.

Time complexity: O(m*n*log(n))
Space complexity: O(1)

*/
public class Solution {
    public int LargestSubmatrix(int[][] matrix) {
        int m = matrix.Length;
    int n = matrix[0].Length;
    for (int j = 0; j < n; ++j)
      for (int i = 1; i < m; ++i)      
        if (matrix[i][j] > 0) matrix[i][j] += matrix[i - 1][j];          
    
    int ans = 0;
    for (int i = 0; i < m; ++i) {
      Array.Sort(matrix[i], (a, b) => b - a);
      for (int j = 0; j < n; ++j)
        ans = Math.Max(ans, (j + 1) * matrix[i][j]);        
    }
    return ans;    
    }
}

// 1732. Find the Highest Altitude
/*Solution: Running Max
Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int LargestAltitude(int[] gain) {
         int ans = 0;
    int cur = 0;
    foreach (int diff in gain)
      ans = Math.Max(ans, cur += diff);
    return ans;
    }
}

// 1733. Minimum Number of People to Teach
/*Solution: Brute Force
Enumerate all languages and see which one is the best.

If two friends speak a common language, we can skip counting them.

Time complexity: O(m*(n+|friendship|))
Space complexity: O(m*n)*/
public class Solution {
    public int MinimumTeachings(int n, int[][] languages, int[][] friendships) {
       
    int m = languages.Length;
    List<HashSet<int>> langs = new List<HashSet<int>>(languages.Length + 1);
    foreach (var l in languages) {
      Array.Sort(l);
      langs.Add(l.ToHashSet());
    }
    foreach (var e in friendships)  {
     HashSet<int> l0 = languages[--e[0]].ToHashSet();
      HashSet<int> l1 = languages[--e[1]].ToHashSet();
         l0.IntersectWith(l1);
      HashSet<int> common = l0;
      //set_intersection(begin(l0), end(l0), begin(l1), end(l1), back_inserter(common));
      if (common.Count > 0) e[0] = e[1] = -1;
    }
    int ans = Int32.MaxValue;
    for (int i = 1; i <= n; ++i) {
      int[] users = new int[m];
      foreach (var e in friendships) {
         // e[0] and e[1] have a common language, skip this edge
        if (e[0] == -1) continue;
        if (!langs[e[0]].Contains(i)) users[e[0]] = 1;
        if (!langs[e[1]].Contains(i)) users[e[1]] = 1;
      }
      ans = Math.Min(ans, users.Sum());      
    }
    return ans;
    }
}

// 1734. Decode XORed Permutation
/*Solution: XOR
The key is to find p[0]. p[i] = p[i – 1] ^ encoded[i – 1]

p[0] ^ p[1] ^ … ^ p[n-1] = 1 ^ 2 ^ … ^ n
encoded[1] ^ encode[3] ^ … ^ encoded[n-2] = (p[1] ^ p[2]) ^ (p[3] ^ p[4]) ^ … ^ (p[n-2] ^ p[n-1])
1) xor 2) = p[0]

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int[] Decode(int[] encoded) {
        int n = encoded.Length + 1;
    int[] perm = new int[n];
    // p[0] = (p[0]^p[1]^...^p[n-1] = 1^2^...^n) 
    //      ^ (p[1]^p[2]^...^p[n-1])
    for (int i = 1; i <= n; ++i) 
      perm[0] ^= i;
    for (int i = 1; i < n; i += 2)
      perm[0] ^= encoded[i];
    for (int i = 1; i < n; ++i)
      perm[i] = perm[i - 1] ^ encoded[i - 1];
    return perm; 
    }
}

// 1736. Latest Time by Replacing Hidden Digits
/*Solution 2: Rules
Using rules, fill from left to right.

Time complexity: O(1)
Space complexity: O(1)*/
public class Solution {
    public string MaximumTime(string time) {
        StringBuilder time1 = new StringBuilder(time);
        if (time1[0] == '?') time1[0] = (time[1] >= '4' && time1[1] <= '9') ? '1' : '2';
    if (time1[1] == '?') time1[1] = (time1[0] == '2') ? '3' : '9';
    if (time1[3] == '?') time1[3] = '5';
    if (time1[4] == '?') time1[4] = '9';
    return time1.ToString();
    }
}

// 1737. Change Minimum Characters to Satisfy One of Three Conditions
/*Clean Solution

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
Space O(26)*/
public class Solution {
    public int MinCharacters(string a, string b) {
        
        
        int m = a.Length, n = b.Length, res = m + n;
        int[] c1 = new int[26], c2 = new int[26];
        for (int i = 0; i < m; ++i)
            c1[a[i] - 'a']++;
        for (int i = 0; i < n; ++i)
            c2[b[i] - 'a']++;

        for (int i = 0; i < 26; ++i) {
            res = Math.Min(res, m + n - c1[i] - c2[i]); // condition 3
            if (i > 0) {
                c1[i] += c1[i - 1];
                c2[i] += c2[i - 1];
            }
            if (i < 25) {
                res = Math.Min(res, m - c1[i] + c2[i]); // condition 1
                res = Math.Min(res, n - c2[i] + c1[i]); // condition 2
            }
        }
        return res;
    }
}

// 1738. Find Kth Largest XOR Coordinate Value
/*Solution: DP
Similar to 花花酱 LeetCode 304. Range Sum Query 2D – Immutable

xor[i][j] = matrix[i][j] ^ xor[i – 1][j – 1] ^ xor[i – 1][j] ^ xor[i][j- 1]

Time complexity: O(mn)
Space complexity: O(mn)*/
public class Solution {
    public int KthLargestValue(int[][] matrix, int k) {
        int m = matrix.Length, n = matrix[0].Length;
    List<int> v = new List<int>();
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        v.Add(matrix[i][j] 
                      ^= (i > 0 ? matrix[i - 1][j] : 0) 
                       ^ (j > 0 ? matrix[i][j - 1] : 0) 
                       ^ (i * j > 0 ? matrix[i - 1][j - 1] : 0));
    //nth_element(begin(v), begin(v) + k - 1, end(v), greater<int>());   
    
    //return v[k - 1];
        
        return v.OrderByDescending(v => v).ElementAt(0 + k - 1);
      
    }
}

// 1739. Building Boxes
/*Solution: Geometry
Step 1: Build a largest pyramid that has less then n cubes, 
whose base area is d*(d+1) / 2
Step 2: Build a largest triangle with cubes left, 
whose base area is l, l*(l + 1) / 2 >= left

Time complexity: O(n^(1/3))
Space complexity: O(1)

*/
public class Solution {
    public int MinimumBoxes(int n) {
        int d = 0;
    int l = 0;
    while (n - (d + 1) * (d + 2) / 2 > 0) {
      n -= (d + 1) * (d + 2) / 2;
      ++d;
    }
    while (n > 0) n -= ++l;
    return d * (d + 1) / 2 + l;
    }
}

// 1742. Maximum Number of Balls in a Box
/*Solution: Hashtable and base-10
Max sum will be 9+9+9+9+9 = 45

Time complexity: O((hi-lo) * log(hi))
Space complexity: O(1)*/
public class Solution {
    public int CountBalls(int lowLimit, int highLimit) {
        int[] balls= new int[46];
    int ans = 0;
    for (int i = lowLimit; i <= highLimit; ++i) {
      int n = i;
      int box = 0;
      while (n > 0) { box += n % 10; n /= 10; }
      ans = Math.Max(ans, ++balls[box]);
    }
    return ans;
    }
}

// 1743. Restore the Array From Adjacent Pairs
/*Solution: Hashtable
Reverse thinking! For a given input array, e.g.
[1, 2, 3, 4, 5]
it’s adjacent pairs are [1,2] , [2,3], [3,4], [4,5]
all numbers appeared exactly twice except 1 and 5, 
since they are on the boundary.
We just need to find the head or tail of the input array, 
and construct the rest of the array in order.

Time complexity:O(n)
Space complexity: O(n)*/
public class Solution {
    public int[] RestoreArray(int[][] adjacentPairs) {
        int n = adjacentPairs.Length + 1;
    Dictionary<int, List<int>> g = new Dictionary<int, List<int>>();
    //for (var i = 0; i < n; i++) g[i] = g.GetValueOrDefault(i,new List<int>());
    foreach (var p in adjacentPairs) {
        g[p[0]] = g.GetValueOrDefault(p[0],new List<int>());
         g[p[1]] = g.GetValueOrDefault(p[1],new List<int>());
      g[p[0]].Add(p[1]);
      g[p[1]].Add(p[0]);
    }
    
    int[] ans = new int[n];
    foreach (var (u, vs) in g)
      if (vs.Count == 1) {
        ans[0] = u;
        ans[1] = vs[0];
        break;
      }
        
    for (int i = 2; i < n; ++i) {
        if(g.ContainsKey(ans[i - 1])){
            var vs = g[ans[i - 1]];
      ans[i] = vs[0] == ans[i - 2] ? vs[1] : vs[0];   
        }
         
    }
    return ans;
    }
}

// 1744. Can You Eat Your Favorite Candy on Your Favorite Day?
/*Solution: Prefix Sum
We must have enough capacity to eat all candies before the current type.
We must have at least prefix sum candies than days, 
since we have to eat at least one each day.
sum[i] = sum(candyCount[0~i])
ans = {days * cap > sum[type – 1] && days <= sum[type])

Time complexity:O(n)
Space complexity: O(n)*/
public class Solution {
    public bool[] CanEat(int[] candiesCount, int[][] queries) {
        int n = candiesCount.Length;
    long[] sums = new long[n + 1];    
    for (int i = 1; i <= n; ++i)
      sums[i] += sums[i - 1] + candiesCount[i - 1];
    List<bool> ans = new List<bool>();
    foreach (var q in queries) {
      long type = q[0], days = q[1] + 1, cap = q[2];
      ans.Add(days * cap > sums[type] && days <= sums[type + 1]);
    }
    return ans.ToArray();
    }
}

// 1745. Palindrome Partitioning IV
/*Solution: DP


dp[i][j] := whether s[i]~s[j] is a palindrome.

dp[i][j] = s[i] == s[j] and dp[i+1][j-1]

ans = any(dp[0][i-1] and dp[i][j] and dp[j][n-1]) for j in range(i, n – 1) for i in range(1, n)

Time complexity: O(n^2)
Space complexity: O(n^2)*/
public class Solution {
    public bool CheckPartitioning(string s) {
        int n = s.Length;
    int[][] dp = new int[n][];//n, vector<int>(n, 1));    
        for(int i = 0; i < n; i++){
             dp[i] = new int[n];Array.Fill(dp[i],1);
        }
           
    for (int l = 2; l <= n; ++l)
      for (int i = 0, j = i + l - 1; j < n; ++i, ++j)
        dp[i][j] = Convert.ToInt32(s[i] == s[j] && dp[i + 1][j - 1] > 0);
    for (int i = 1; i < n; ++i)
      for (int j = i; j + 1 < n; ++j)
        if (dp[0][i - 1] > 0 && dp[i][j] > 0 && dp[j + 1][n - 1] > 0)
          return true;
    return false;
    }
}

// 1748. Sum of Unique Elements
/*Solution: Hashtable
Time complexity: O(n)
Space complexity: O(100)

*/
public class Solution {
    public int SumOfUnique(int[] nums) {
         int[] seen = new int[101];
    int ans = 0;
    foreach (int x in nums)
      ++seen[x];
    foreach (int x in nums)
      if (seen[x] == 1) ans += x;
    return ans;
    }
}

// 1749. Maximum Absolute Sum of Any Subarray
/*Solution: Prefix Sum
ans = max{abs(prefix_sum[i] – max(prefix_sum[0:i])), abs(prefix_sum – min(prefix_sum[0:i])}

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int MaxAbsoluteSum(int[] nums) {
        int lo = 0;
    int hi = 0;
    int s = 0;
    int ans = 0;
    foreach (int x in nums) {
      s += x;
      ans = Math.Max(Math.Max(ans, Math.Abs(s - lo)), Math.Abs(s - hi));
      hi = Math.Max(hi, s);
      lo = Math.Min(lo, s);
    }
    return ans;
    }
}

// 1750. Minimum Length of String After Deleting Similar Ends
/*Solution: Two Pointers + Greedy
Delete all the chars for each prefix and suffix pair.

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int MinimumLength(string s) {
        int l = 0, r = s.Length - 1;
    while (l < r) {
      if (s[l] != s[r]) break;
       char c = s[l];
      while (l <= r && s[l] == c) ++l;
      while (l <= r && s[r] == c) --r;
    }
    return r - l + 1;
    }
}

// 1751. Maximum Number of Events That Can Be Attended II
/*Solution: DP + Binary Search
Sort events by ending time.
dp[i][j] := max value we can get by attending at most j events among events[0~i].
dp[i][j] = max(dp[i – 1][j], dp[p][j – 1] + value[i])
p is the first event that does not overlap with the current one.

Time complexity: O(nlogn + nk)
Space complexity: O(nk)*/
public class Solution {
    public int MaxValue(int[][] events, int k) {
        int n = events.Length; 
    int[][] dp = new int[n + 1][];//(n + 1, vector<int>(k + 1)); 
        for(int i = 0; i < n + 1; i++){
            dp[i] = new int[k + 1];Array.Fill(dp[i],0);
        }
            
       // int[] e = new int[n];
       // for (int i = 1; i <= n; i++) e[i-1] = binarySearsh(events, events[i - 1][0]);
   /* auto comp(const vector<int>& a, const vector<int>& b) {
      return a[1] < b[1];
    };*/
    
    Array.Sort(events, (a, b) => a[1]-b[1]);

    for (int i = 1; i <= n; ++i) {
     // int p = lower_bound(begin(events),  begin(events) + i, int[3]{0, events[i - 1][0], 0}, comp) - begin(events);
         int p =  binearySearch(events, events[i-1][0]);

      for (int j = 1; j <= k; ++j)
        dp[i][j] = Math.Max(dp[i - 1][j], 
                       dp[p][j - 1] + events[i - 1][2]);
    }    
    return dp[n][k];
    }
    
   private int binearySearch(int[][] events, int key) {
        int l = 0, r = events.Length;
        while(l < r) {
            int m = (r-l)/2 + l;
            if(events[m][1] < key) {
                l = m+1;
            }else{
                r = m;
            }
        }
        
        return l;
    }
}
// Bottom Up DP Solution
public class Solution {
    public int MaxValue(int[][] events, int k) {
        Array.Sort(events, (a, b)=> a[1] - b[1]);
        int[][] dp = new int[events.Length+1][];
         for(int i = 0; i < events.Length+1; i++) 
         dp[i] = new int[k+1];
        for(int i = 1; i <= k; i++) {
            for(int j = 1; j <= events.Length; j++) {
                dp[j][i] = Math.Max(dp[j][i-1], dp[j-1][i]);
                int m = binearySearch(events, events[j-1][0]);
                dp[j][i] = Math.Max(dp[m][i-1] + events[j-1][2], dp[j][i]);
            }
        }
        
        return dp[events.Length][k];
    }
    
    private int binearySearch(int[][] events, int key) {
        int l = 0, r = events.Length;
        while(l < r) {
            int m = (r-l)/2 + l;
            if(events[m][1] < key) {
                l = m+1;
            }else{
                r = m;
            }
        }
        
        return l;
    }
    
}
// Bottom Up DP Solution
/*Intuition:

Let DP[i][j] := Maximum value attending at mostjevents among the firstith events

With this definition, our goal is to calculate DP[N][K].

We cannot blindly attend both i - 1th and ith events because there is possiblity of confliction.
Therefore, we need to pre-construct prev[i] := the index of valid previous event of i-th event to look up the valid previous event.

The transition of DP is

// the maximum value by attending at most j events can be the same 
// as what achieved by attending at most j - 1 events.
dp[i][j]= Math.max(dp[i][j], dp[i][j-1]);

// the maximum value by attending j events from 0 to i-th events can be the same
// as what achieved by attending at most the same number of events from the first 0 to (i-1)-th events
dp[i][j] = Math.max(dp[i][j], dp[i-1][j]);

// attend i-th event. In this case, add  event value (= events[i][2]) to dp[prev[i]][j-1]
dp[i][j] = Math.max(dp[i][j], dp[prev[i]][j-1] + events[i][2]); */
public class Solution {
    public int MaxValue(int[][] events, int k) {
        int n = events.Length;
        Array.Sort(events, (o1, o2) => o1[1] - o2[1]);

        int[] prev = new int[n];
        int[][] dp = new int[n+1][];
        
        for (int i = 0; i < n; i++) {
            prev[i] = binarySearsh(events, events[i][0]);
            dp[i] = new int[k+1];Array.Fill(dp[i], 0);
        } 
         dp[n] = new int[k+1];Array.Fill(dp[n], 0);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                dp[i+1][j+1] = Math.Max(dp[i+1][j+1], dp[i+1][j]);
                dp[i+1][j+1] = Math.Max(dp[i+1][j+1], dp[i][j+1]);
                dp[i+1][j+1] = Math.Max(dp[i+1][j+1], dp[prev[i] + 1][j] + events[i][2]);
            }
        }

        return dp[n][k];
    }

    private int binarySearsh(int[][] a, int x) {//BinarySearch for matrix
        int l = -1, r = a.Length;
        while (r - l > 1) {
            int m = (l + r) / 2;
            if (a[m][1] < x) l = m;
            else r = m;
        }
        return l;
    }
    
}
//Will Cause TLE. 57 / 67 test cases passed.
public class Solution {
    public int MaxValue(int[][] events, int k) {
        int n = events.Length; 
    int[][] dp = new int[n + 1][];//(n + 1, vector<int>(k + 1)); 
        for(int i = 0; i < n + 1; i++){
            dp[i] = new int[k + 1]; Array.Fill(dp[i],0);
        }
    
    Array.Sort(events, (a, b) => a[1]==b[1]? a[0]-b[0] : a[1]-b[1]);
        // a[1]==b[1]? a[0]-b[0] : a[1]-b[1]
    for (int i = 1; i <= n; ++i) {
     // int p = binarySearch(events, events[i - 1][1], i);//lower_bound(begin(events),  begin(events) + i, int[3]{0, events[i - 1][0], 0}, comp) - begin(events);
        int p = events.ToList().BinarySearch(new int[3]{0, events[i - 1][0], 0}, Comparer<int[]>.Create((a, b) =>  a[1]==b[1]? a[0]-b[0] : a[1]-b[1]) );
        if (p < 0) p = ~p;
      for (int j = 1; j <= k; ++j)
        dp[i][j] = Math.Max(dp[i - 1][j], 
                       dp[p][j - 1] + events[i - 1][2]);
    }    
    return dp[n][k];
    }
    
}
// 1752. Check if Array Is Sorted and Rotated
/*Easy and Concise

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
Space O(1)*/
public class Solution {
    public bool Check(int[] nums) {
       int k = 0, n = nums.Length;
        for (int i = 0; i < n; ++i) {
            if (nums[i] > nums[(i + 1) % n]) {
                k++;
            }
            if (k > 1) {
                return false;
            }
        }
        return true;
    }
}
/*Solution: Counting and checking
Count how many turning points (nums[i] < nums[i – 1]) in the array. 
Return false if there are more than 1.
For the turning point r, (nums[r] < nums[r – 1), 
return true if both of the following conditions are satisfied:
1. nums[r – 1] is the largest number, e.g. nums[r – 1] >= nums[n – 1]
2. nums[r] is the smallest number, e.g. nums[r] <= nums[0]

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public bool Check(int[] nums) {
 int n = nums.Length, dropPoint = 0;
        for (int i = 1; i < n; i++) {
            if (nums[i] < nums[i - 1]) dropPoint++;
        }
        if (dropPoint == 0) return true;
        if (dropPoint == 1 && nums[0] >= nums[n - 1]) return true;
        return false;
    }
}

// 1753. Maximum Score From Removing Stones
/*Solution 2: Math
First, let’s assuming a <= b <= c.
There are two conditions:
1. a + b <= c, we can pair c with a first and then b. 
Total pairs is (a + b + (a + b)) / 2
2. a + b > c, we can pair c with a, b “evenly”, 
and then pair a with b, total pairs is (a + b + c) / 2

ans = (a + b + min(a + b, c)) / 2

Time complexity: O(1)
Space complexity: O(1)*/
public class Solution {
    public int MaximumScore(int a, int b, int c) {
        int[] s = new int[3]{a,b,c};
    Array.Sort(s);
    return (s[0] + s[1] + Math.Min(s[0] + s[1], s[2])) / 2;
    }
}

// 1754. Largest Merge Of Two Strings
/*Solution: Greedy
Always take a single char from the largest word. (NOT just the current char).

E.g.
ans = “”, w1 = “cabba”, w2 = “bcaaa”
w1 > w2, take from w1
ans = “c”, w1 = “abba”, w2 = “bcaaa”
w1 < w2, take from w2
ans = “cb”, w1 = “abba”, w2 = “caaa”
w1 < w2, take from w2
ans = “cbc”, w1 = “abba”, w2 = “aaa”
w1 > w2, take from w1. 
Note: both start with “a”, but we need to compare the entire word.
ans = “cbca”, w1 = “bba”, w2 = “aaa”
w1 > w2, take from w1
ans = “cbcab”, w1 = “ba”, w2 = “aaa”
…

Time complexity: O(min(m,n)^2)
Space complexity: O(1)

*/
public class Solution {
    public string LargestMerge(string word1, string word2) {
        StringBuilder ans = new StringBuilder();    
    int m = word1.Length, n = word2.Length;
    int i = 0, j = 0;
    while (i < m && j < n)    
        //word1.Substring(i) > word2.Substring(j) ?
      if (string.Compare(word1.Substring(i), word2.Substring(j)) > 0 ) 
          ans.Append(word1[i++]); 
        else ans.Append(word2[j++]);
    ans.Append(word1.Substring(i));
    ans.Append(word2.Substring(j));
    return ans.ToString();
    }
}

// 1755. Closest Subsequence Sum
/*Solution: Binary Search

Since n is too large to generate sums for all subsets O(2^n), 
we have to split the array into half, generate two sum sets. O(2^(n/2)).

Then the problem can be reduced to find the closet sum by picking one number (sum) 
each from two different arrays which can be solved in O(mlogm), where m = 2^(n/2).

So final time complexity is O(n * 2^(n/2))
Space complexity: O(2^(n/2))

*/
public class Solution {
    public int MinAbsDifference(int[] nums, int goal) {
         int n = nums.Length;
    int ans = Math.Abs(goal);
    List<int> t1 = new List<int>(1 << (n / 2 + 1)), t2 = new List<int>(1 << (n / 2 + 1));t1.Add(0);t2.Add(0);
   // t1.reserve(1 << (n / 2 + 1));
   // t2.reserve(1 << (n / 2 + 1));
    for (int i = 0; i < n / 2; ++i)
      for (int j = t1.Count - 1; j >= 0; --j)
        t1.Add(t1[j] + nums[i]);
    for (int i = n / 2; i < n; ++i)
      for (int j = t2.Count - 1; j >= 0; --j)
        t2.Add(t2[j] + nums[i]);
    List<int> s2 = t2.ToHashSet().OrderBy(x => x).ToList();
       //  Console.WriteLine("t2 : " + String.Join(',', t2));
      //  Console.WriteLine("t1 : " + String.Join(',', t1));
    foreach (int x in t1.ToHashSet()) {
      int it = lowerBound(s2, goal - x);
      if (it != s2.Count)
        ans = Math.Min(ans, Math.Abs(goal - x - s2[it]));
      if (it != 0)
        ans = Math.Min(ans, Math.Abs(goal - x - s2[it - 1]));

    }
    return ans;
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
}

// 1758. Minimum Changes To Make Alternating Binary String
/*Solution: Two Counters
The final string is either 010101… or 101010…
We just need two counters to record the number of changes 
needed to transform the original string to those two final strings.

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int MinOperations(string s) {
        int c1 = 0, c2 = 0;
    for (int i = 0; i < s.Length; ++i) {
      c1 += Convert.ToInt32(s[i] - '0' == i % 2);
      c2 += Convert.ToInt32(s[i] - '0' != i % 2);
    }
    return Math.Min(c1, c2);
    }
}

// 1759. Count Number of Homogenous Substrings
/*Solution: Counting

Let m be the length of the longest homogenous substring, 
# of homogenous substring is m * (m + 1) / 2.
e.g. aaabb
“aaa” => m = 3, # = 3 * (3 + 1) / 2 = 6
“bb” => m = 2, # = 2 * (2+1) / 2 = 3
Total = 6 + 3 = 9

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int CountHomogenous(string s) {
        int kMod = (int)1e9 + 7;
    int n = s.Length;
    long ans = 0;
    for (long i = 0, j = 0; i < n; i = j) {
      while (j < n && s[(int)i] == s[(int)j]) ++j;
      ans += (j - i) * (j - i + 1) / 2;
    }
    return Convert.ToInt32(ans % kMod);
    }
}

// 1700. Number of Students Unable to Eat Lunch
/*Solution 2: Counting
Count student’s preferences. Then process students from 1 to n, 
if there is no sandwich for current student then we can stop, 
since he/she will block all the students behind him/her.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int CountStudents(int[] students, int[] sandwiches) {
        int n = students.Length;
    int[] c = new int[2];
    foreach (int p in students) ++c[p];
    for (int i = 0; i < n; ++i)
      if (--c[sandwiches[i]] < 0) return n - i;
    return 0;
    }
}

// 1701. Average Waiting Time
/*When a customer arrives, if the arrival time is greater than current, 
then advance the clock to arrival time. 
Advance the clock by cooking time. Waiting time = current time - arrival time.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public double AverageWaitingTime(int[][] customers) {
        int t = 0;
    double w = 0;
    foreach (var c in customers) {
      if (c[0] > t) t = c[0];
      t += c[1];
      w += t - c[0];
    }
    return w / customers.Length;
    }
}

// 1702. Maximum Binary String After Change
/*Solution with Explanation

Explanation
We don't need touch the starting 1s, they are already good.

For the rest part,
we continually take operation 2,
making the string like 00...00011...11

Then we continually take operation 1,
making the string like 11...11011...11.


Complexity
Time O(n)
Space O(n)*/
public class Solution {
    public string MaximumBinaryString(string binary) {
        int ones = 0, zeros = 0, n = binary.Length;
        StringBuilder res = new StringBuilder(n).Insert(0, "1", n);
        for (int i = 0; i < n; ++i) {
            if (binary[i] == '0')
                zeros++;
            else if (zeros == 0)
                ones++;
        }
        if (ones < n)
            res[ones + zeros - 1] =  '0';
        return res.ToString();
    }
}

// 1703. Minimum Adjacent Swaps for K Consecutive Ones
/*Solution: Prefix Sum + Sliding Window
Time complexity: O(n)
Space complexity: O(n)

We only care positions of 1s, we can move one element from position x to y 
(assuming x + 1 ~ y are all zeros) in y – x steps. 
e.g. [0 0 1 0 0 0 1] => [0 0 0 0 0 1 1], 
move first 1 at position 2 to position 5, cost is 5 – 2 = 3.

Given a size k window of indices of ones, 
the optimal solution it to use the median number as center. 
We can compute the cost to form consecutive numbers:

e.g. [1 4 7 9 10] => [5 6 7 8 9] cost = (5 – 1) + (6 – 4) + (9 – 8) + (10 – 9) = 8

However, naive solution takes O(n*k) => TLE.

We can use prefix sum to compute the cost of a window in O(1) to 
reduce time complexity to O(n)

First, in order to use sliding window, 
we change the target of every number in the window to the median number.
e.g. [1 4 7 9 10] => [7 7 7 7 7] cost = (7 – 1) + (7 – 4) + (7 – 7) + (9 – 7) + (10 – 7) = (9 + 10) – (1 + 4) = right – left.
[5 6 7 8 9] => [7 7 7 7 7] takes extra 2 + 1 + 1 + 2 = 6 steps = (k / 2) * ((k + 1) / 2), 
these extra steps should be deducted from the final answer.*/
public class Solution {
    public int MinMoves(int[] nums, int k) {
        List<long> s = new List<long>(); s.Add(1);
    for (int i = 0; i < nums.Length; ++i)
      if (nums[i] != 0)
        s.Add(s.Last() + i);
    int n = s.Count;
    long ans = (long)1e10;
    int m1 = k / 2, m2 = (k + 1) / 2;
    for (int i = 0; i + k < n; ++i) {
      long right = s[i + k] - s[i + m1];
      long left = s[i + m2] - s[i];
      ans = Math.Min(ans, right - left);
    }
    return Convert.ToInt32(ans - m1 * m2);
    }
}

// 1704. Determine if String Halves Are Alike

// 1705. Maximum Number of Eaten Apples
/*Solution: PriorityQueue
Sort by rotten day in ascending order, 
only push onto the queue when that day has come (be able to grow apples).

Time complexity: O((n+ d)logn)
Space complexity: O(n)*/
public class Solution {
    public int EatenApples(int[] apples, int[] days) {
        int n = apples.Length;
    //using P = pair<int, int>;    
    //PriorityQueue<KeyValuePair<int, int>, vector<P>, greater<P>> q; // {rotten_day, index}    
        PriorityQueue<KeyValuePair<int, int>, KeyValuePair<int, int>> q = new PriorityQueue<KeyValuePair<int, int>, KeyValuePair<int, int>>(Comparer<KeyValuePair<int,int>>.Create((a, b) => a.Key - b.Key));
    int ans = 0;
    for (int d = 0; d < n || q.Count != 0; ++d) {
      if (d < n && apples[d] != 0) q.Enqueue(new KeyValuePair<int, int>(d + days[d], d), new KeyValuePair<int, int>(d + days[d], d));//q.emplace(d + days[d], d);
      while (q.Count != 0 
             && (q.Peek().Key <= d || apples[q.Peek().Value] == 0)) q.Dequeue();
      if (q.Count == 0 ) continue;
      --apples[q.Peek().Value];      
      ++ans;
    }
    return ans;
    }
}

// 1711. Count Good Meals
/*Solution: Hashtable
Same idea as LeetCode 1: Two Sum

Use a hashtable to store the occurrences of all the numbers added so far. 
For a new number x, check all possible 2^i – x. ans += freq[2^i – x] 0 <= i <= 21

Time complexity: O(22n)
Space complexity: O(n)*/
public class Solution {
    public int CountPairs(int[] deliciousness) {
        int kMod = (int)1e9 + 7;
    Dictionary<int, int> m = new Dictionary<int, int>();    
    long ans = 0;
    foreach (int x in deliciousness) {
      for (int t = 1; t <= 1 << 21; t *= 2) {
        if( m.ContainsKey(t - x)) 
            ans += m[t - x];
        //if (it != end(m)) ans += it->second;
      }
      //++m[x];
        m[x] = m.GetValueOrDefault(x, 0) + 1;
    }
    return Convert.ToInt32(ans % kMod);
    }
}

// 1710. Maximum Units on a Truck
/*Solution: Greedy
Sort by unit in descending order.

Time complexity: O(nlogn)
Space complexity: O(1)

*/
public class Solution {
    public int MaximumUnits(int[][] boxTypes, int truckSize) {
     /*   sort(begin(boxTypes), end(boxTypes), [](const auto& a, const auto& b){
      return a[1] > b[1]; // Sort by unit DESC
    });*/
    
    Array.Sort(boxTypes, (a,b) => b[1] - a[1]);
    int ans = 0;
    foreach (var b in boxTypes) {      
      ans += b[1] * Math.Min(b[0], truckSize);      
      if ((truckSize -= b[0]) <= 0) break;      
    }
    return ans;
    }
}

// 1716. Calculate Money in Leetcode Bank
/*Solution 1: Simulation
Increase the amount by 1 everyday, the decrease 6 after every sunday.

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int TotalMoney(int n) {
        int total = 0;
    for (int d = 0, m = 1; d < n; ++d) {
      total += m++;
      if (d % 7 == 6) m -= 6;
    }
    return total;
    }
}

// 1717. Maximum Score From Removing Substrings
/*Solution: Greedy + Stack
Remove the pattern with the larger score first.

Using a stack to remove all occurrences of a pattern in place in O(n) Time.

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int MaximumGain(string s, int x, int y) {
        // Remove patttern p from s for t points each.
    // Returns the total score.
    StringBuilder sb = new StringBuilder(s);
        
        int remove(String p, int t) {
        int i = 0, total = 0;
        for (int j = 0; j < sb.Length; ++j, ++i) {
            sb[i] = sb[j];
            if (i > 0 && sb[i-1] == p[0] && sb[i] == p[1]) {
                i -= 2;
                total += t;
            }
        }
        sb.Length = i;
        return total;
    };
        if (x > y) {
            return remove("ab", x) + remove("ba", y);
        }
        return remove("ba", y) + remove( "ab", x);
    
    }
}

// 1718. Construct the Lexicographically Largest Valid Sequence
/*Solution: Search
Search from left to right, largest to smallest.

Time complexity: O(n!)?
Space complexity: O(n)*/
public class Solution {
    public int[] ConstructDistancedSequence(int n) {

    int[] ans = new int[n * 2 - 1];
    dfs(ans, n, 0, 0);
    return ans;
    }
     private bool dfs(int[] ans, int n, int i, int s) {
     if (i == ans.Length) return true;
    if (ans[i] > 0) return dfs(ans, n, i + 1, s);
    for (int d = n; d > 0; --d) {
      int j = i + (d == 1 ? 0 : d);
      if ((s & (1 << d)) > 0 || j >= ans.Length || ans[j] > 0)
        continue;
      ans[i] = ans[j] = d;
      if (dfs(ans, n, i + 1, s | (1 << d))) return true;
      ans[i] = ans[j] = 0;
    }
    return false;
  }
  
}

// 1719. Number Of Ways To Reconstruct A Tree
/*Solution: Bitset
Time complexity: O(E*V)
Space complexity: O(V^2)

*/

// 1720. Decode XORed Array
/*Solution: XOR
encoded[i] = arr[i] ^ arr[i + 1]
encoded[i] ^ arr[i] = arr[i] ^ arr[i] ^ arr[i + 1]
arr[i+1] = encoded[i]^arr[i]

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int[] Decode(int[] encoded, int first) {
        int n = encoded.Length + 1;
    int[] ans = new int[n];Array.Fill(ans, first);
    for (int i = 0; i + 1 < n; ++i)
      ans[i + 1] = ans[i] ^ encoded[i];
    return ans;
    }
}

// 1721. Swapping Nodes in a Linked List
/*Solution:
Two passes. 
First pass, find the length of the list. 
Second pass, record the k-th and n-k+1-th node.
Once done swap their values.

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
    public ListNode SwapNodes(ListNode head, int k) {
         int l = 0;
    ListNode cur = head;    
    while (cur != null) { cur = cur.next; ++l; }    
    
    cur = head;
    ListNode n1 = null;
    ListNode n2 = null;
    for (int i = 1; i <= l; ++i, cur = cur.next) {
      if (i == k) n1 = cur;
      if (i == l - k + 1) n2 = cur;
    }
    //swap(n1.val, n2.val);
        int t = n1.val;
        n1.val = n2.val;
        n2.val = t;
    return head;
    }
}

// 1722. Minimize Hamming Distance After Swap Operations
/*Solution: Union Find
Similar to 花花酱 LeetCode 1202. Smallest String With Swaps

Think each pair as an edge in a graph. 
Since we can swap as many time as we want, 
which means we can arrange the elements whose indices are 
in a connected component (CC) in any order.

For each index i, we increase the counter of CC(i) for key source[i] 
and decrease the counter of the same CC for key target[i]. 
If two keys are the same (can from different indices), 
one from source and one from target, it will cancel out, no distance. 
Otherwise, the counter will be off by two. 
Finally we sum up the counter for all the keys and divide it by two to 
get the hamming distance.

Time complexity: O(V+E)
Space complexity: O(V)*/
public class Solution {
    public int MinimumHammingDistance(int[] source, int[] target, int[][] allowedSwaps) {
         int n = source.Length;
    int[] p = Enumerable.Range(0, n).ToArray();
    //iota(begin(p), end(p), 0);
    
    int find(int x) {
      return x == p[x] ? x : p[x] = find(p[x]);
    };
    
    foreach (var s in allowedSwaps)
      p[find(s[0])] = find(s[1]);
    
   Dictionary<int, Dictionary<int, int>> m = new Dictionary<int, Dictionary<int, int>>();
    for (int i = 0; i < n; ++i) {
     // ++m[find(i)][source[i]];
        m[find(i)] = m.GetValueOrDefault(find(i) ,new Dictionary<int,int>() );
        m[find(i)][source[i]] =  m[find(i)].GetValueOrDefault(source[i], 0)+1;
        m[find(i)] = m.GetValueOrDefault(find(i) ,new Dictionary<int,int>() );
        m[find(i)][target[i]] =  m[find(i)].GetValueOrDefault(target[i], 0)-1;
      //--m[find(i)][target[i]];
    }
    
    int ans = 0;
    foreach (var g in m) 
      foreach (var kv in g.Value)
        ans += Math.Abs(kv.Value);
    return ans / 2;
    }
}

// 1723. Find Minimum Time to Finish All Jobs
/*Solution 2: Search + Pruning
Time complexity: O(k^n)
Space complexity: O(k*n)

*/
public class Solution {

    public int MinimumTimeRequired(int[] jobs, int k) {
    int[] times = new int[k];
    int ans = Int32.MaxValue;
    void dfs (int i, int cur) {
      if (cur >= ans) return;
      if (i == jobs.Length ) {
        ans = cur;
        return;
      }
      HashSet<int> seen = new HashSet<int>();
      for (int j = 0; j < k; ++j) {
          
        if(!seen.Add(times[j])) continue;
        times[j] += jobs[i];
        dfs(i + 1, Math.Max(cur, times[j]));
        times[j] -= jobs[i];
      }
    };
    Array.Sort(jobs, (a, b) => b - a);
    dfs(0, 0);
    return ans;
    }
}
// DFS
/*Solution 2: Bianry search
The problem of the first solution,
is that the upper bound reduce not quick enough.
Apply binary search, to reduce the upper bound more quickly.*/
public class Solution {
    public int MinimumTimeRequired(int[] jobs, int k) {
    Array.Sort(jobs);
        int n = jobs.Length;
        int left = jobs[n - 1];
        int right = jobs[n - 1] * n;
        while (left < right) {
            int[] cap = new int[k];
            int mid = left + (right - left) / 2;
            Array.Fill(cap, mid);
            if (dfs(jobs, cap, n - 1, k, mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    private bool dfs(int[] jobs, int[] cap, int i, int k, int x) {
        if (i == -1) {
            return true;
        }
        for (int j = 0; j < k; j++) {
            if (cap[j] >= jobs[i]) {
                cap[j] -= jobs[i];
                if (dfs(jobs, cap, i - 1, k, x)) {
                    return true;
                }
                cap[j] += jobs[i];
            }
            if (cap[j] == x) {
                break;
            }
        }
        return false;
    }
}

public class Solution {
         int result = Int32.MaxValue;
    public int MinimumTimeRequired(int[] jobs, int k) {
     //Arrays.sort(jobs);
        backtracking(jobs, 0, new int[k], 0);
        return result;
    }
    
    private void backtracking(int[] jobs, int index, int[] workers, int max) {
        
        if (index == jobs.Length) {
            result = Math.Min(result, max);
            return;
        }
        
        if (max > result) {
            return;
        }
        
        for (int i = 0; i < workers.Length; i++) {
            workers[i] += jobs[index];
            backtracking(jobs, index+1, workers, Math.Max(workers[i], max));
            workers[i] -= jobs[index];
            
            if (workers[i] == 0) {
                break;
            }
        }
    }
}
// 1725. Number Of Rectangles That Can Form The Largest Square
/*Solution: Running Max of Shortest Edge
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int CountGoodRectangles(int[][] rectangles) {
         int cur = 0;
    int ans = 0;
    foreach (var r in rectangles) {
      if (Math.Min(r[0], r[1]) > cur) {
        cur = Math.Min(r[0], r[1]);
        ans = 0;
      }
      if (Math.Min(r[0], r[1]) == cur) ++ans;
    }
    return ans;
    }
}

// 1726. Tuple with Same Product
/*Solution: HashTable
Similar idea to 花花酱 LeetCode 1. Two Sum

Use a hashtable to store all the pair product counts.

Enumerate all possible pairs, increase the answer by the same product counts * 8.

Why time 8? C(4,1) * C(1,1) * C(2,1) * C(1,1)

For pair one AxB, A can be placed at any position in a four tuple, 
B’s position is then fixed. For another pair CxD, 
C has two positions to choose from, D is fixed.

Time complexity: O(n^2)
Space complexity: O(n^2)*/
public class Solution {
    public int TupleSameProduct(int[] nums) {
       int n = nums.Length;
    Dictionary<int, int> m = new Dictionary<int, int>();
    int ans = 0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < i; ++j){
      //m[nums[i] * nums[j]]++
          ans += 8 * m.GetValueOrDefault( nums[i] * nums[j] , 0);
          m[nums[i] * nums[j]] = m.GetValueOrDefault( nums[i] * nums[j] , 0)+1;
           
      }       
       
    return ans;
    }
}

// 1697. Checking Existence of Edge Length Limited Paths
/*Solution: Union Find
Since queries are offline, we can reorder them to optimize time complexity. 
Answer queries by their limits in ascending order while union edges 
by weights up to the limit. 
In this case, we just need to go through the entire edge list at most once.

Time complexity: O(QlogQ + ElogE)
Space complexity: O(Q + E)*/
public class Solution {
    public bool[] DistanceLimitedPathsExist(int n, int[][] edgeList, int[][] queries) {
       // int[] parents = new int[n];
        int[] parents = Enumerable.Range(0, n + 1).ToArray();
    //iota(begin(parents), end(parents), 0);
    int find (int x) {
      return parents[x] == x ? x : parents[x] = find(parents[x]);
    };
    int m = queries.Length;
   //var t = queries.ToList();
    for (int j = 0; j < m; ++j)  {
        var t = queries[j].ToList();t.Add(j);
        queries[j] = t.ToArray();
    }
 
    // Sort edges by weight in ascending order.
    Array.Sort(edgeList, (a,b) => a[2] - b[2] );
    // Sort queries by limit in ascending order
    Array.Sort(queries, (a,b) => a[2] - b[2] );
    //sort(begin(queries), end(queries), [](const auto& a, const auto& b) { return a[2] - b[2]; });
   bool[] ans = new bool[m];
    int i = 0;
    foreach (var q in queries) {      
      while (i < edgeList.Length && edgeList[i][2] < q[2]){
        parents[find(edgeList[i][0])] = find(edgeList[i][1]);   i++;
      }
               
      ans[q[3]] = find(q[0]) == find(q[1]);
    }
    return ans;
    }
}

// 1696. Jump Game VI
/*Solution: DP + Monotonic Queue
dp[i] = nums[i] + max(dp[j]) i – k <= j < i

Brute force time complexity: O(n*k) => TLE
This problem can be reduced to find the maximum for 
a sliding window that can be solved by monotonic queue.

Time complexity: O(n)
Space complexity: O(n+k) -> O(k)*/
public class Solution {
    public int MaxResult(int[] nums, int k) {
        int n = nums.Length;
    LinkedList<KeyValuePair<int, int>> q = new LinkedList<KeyValuePair<int, int>>();
        q.AddLast(new KeyValuePair<int, int>(nums[0], 0));
    for (int i = 1; i < n; ++i) {      
      int cur = nums[i] + q.First().Key;      
      while (q.Count != 0 && cur >= q.Last().Key) 
        q.RemoveLast();
      while (q.Count != 0 && i - q.First().Value >= k) 
        q.RemoveFirst();    
      q.AddLast(new KeyValuePair<int, int>(cur, i));
    }
    foreach (var (v, i) in q)
      if (i == n - 1) return v;
    return 0;
    }
}

// 1695. Maximum Erasure Value
/*Solution: Sliding window + Hashset
Maintain a window that has no duplicate elements.

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int MaximumUniqueSubarray(int[] nums) {
        int n = nums.Length;
    HashSet<int> t = new HashSet<int>();
    int ans = 0;
    for (int l = 0, r = 0, s = 0; r < n; ++r) {
      while (t.Contains(nums[r]) && l < r) {
        s -= nums[l];
        t.Remove(nums[l++]);
      }      
      t.Add(nums[r]);
      ans = Math.Max(ans, s += nums[r]);
    }
    return ans;
    }
}

// 1694. Reformat Phone Number
/*Solution:
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public string ReformatNumber(string number) {
        StringBuilder ans = new StringBuilder();
        int total = 0;
         foreach (char c in number) {
      if (Char.IsDigit(c)) total += 1;
             }
  /*  int total = count_if(begin(number), end(number), 
                               [](char c) { return isdigit(c); });*/
    int l = 0;
    foreach (char c in number) {
      if (!Char.IsDigit(c)) continue;      
       ans.Append(c);
      ++l;
      if ((l % 3 == 0 && total - l >= 4) ||
          (total % 3 != 0 && total - l == 2)
          || (total % 3 == 0 && total - l == 3))
        ans.Append("-");
    }
    return ans.ToString();
    }
}

// 1691. Maximum Height by Stacking Cuboids
/*Solution: Math/Greedy + DP
Direct DP is very hard, since there is no orders.

We have to find some way to sort the cuboids such 
that cuboid i can NOT stack on cuboid j if i > j. 
Then dp[i] = max(dp[j]) + height[i], 0 <= j < i, 
for each i, find the best base j and stack on top of it.
ans = max(dp)

We can sort the cuboids by their sorted dimensions, 
and cuboid i can stack stack onto cuboid j if and 
only if w[i] <= w[j] and l[i] <= l[j] and h[i] <= h[j].

First of all, we need to prove that all heights must come 
from the largest dimension of each cuboid.

1. If the top of the stack is A1*A2*A3, A3 < max(A1, A2), 
we can easily swap A3 with max(A1, A2), 
it’s still stackable but we get larger heights.
e.g. 3x5x4, base is 3×5, height is 4, 
we can rotate to get base of 3×4 with height of 5.

2. If a middle cuboid A of size A1*A2*A3, assuming A1 >= A2, A1 > A3, 
on top of A we have another cuboid B of size B1*B2*B3, B1 <= B2 <= B3.
We have A1 >= B1, A2 >= B2, A3 >= B3, by rotating A we have A3*A2*A1
A3 >= B3 >= B1, A2 >= B2, A1 > A3 >= B3, so B can be still on top of A, 
and we get larger height.

e.g. A: 3x5x4, B: 2x3x4
A -> 3x4x5, B is still stackable.

…

Time complexity: O(n^2)
Space complexity: O(n^2)*/
public class Solution {
    public int MaxHeight(int[][] cuboids) {
       // Console.WriteLine(cuboids.Length);
      List<int[]> t = cuboids.ToList();
        t.Add(new int[3]{0, 0, 0});
        cuboids = t.ToArray();
        //Console.WriteLine(cuboids.Length);
    int n = cuboids.Length;
    foreach (int[] box in cuboids) Array.Sort(box, (a,b) => a - b);
   Array.Sort(cuboids, (a,b) => a[0] == b[0] ? a[1] == b[1] ? a[2] - b[2] : a[1] - b[1] : a[0] - b[0]);
    int[] dp = new int[n];
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < i; ++j)
        if (cuboids[i][0] >= cuboids[j][0] && cuboids[i][1] >= cuboids[j][1] && cuboids[i][2] >= cuboids[j][2])
          dp[i] = Math.Max(dp[i], dp[j] + cuboids[i][2]);
    return dp.Max();//*max_element(begin(dp), end(dp));
    }
}

// 1690. Stone Game VII
/*Solution: MinMax + DP

For a sub game of stones[l~r] game(l, r), we have two choices:
Remove the left one: sum(stones[l + 1 ~ r]) – game(l + 1, r)
Remove the right one: sum(stones[l ~ r – 1]) – game(l, r – 1)
And take the best choice.

Time complexity: O(n^2)
Space complexity: O(n^2)

Bottom-Up*/
public class Solution {
    public int StoneGameVII(int[] stones) {
        int n = stones.Length;
    int[] s = new int[n + 1];
    for (int i = 0; i < n; ++i) s[i + 1] = s[i] + stones[i];
    //vector<vector<int>> dp(n, vector<int>(n, 0));
        int[][] dp = new int[n][];  
         for(int i = 0; i < n; i++){
             dp[i] = new int[n];Array.Fill(dp[i],0);
         }
            
    for (int c = 2; c <= n; ++c)
      for (int l = 0, r = l + c - 1; r < n; ++l, ++r)
        dp[l][r] = Math.Max(s[r + 1] - s[l + 1] - dp[l + 1][r],
                       s[r] - s[l] - dp[l][r - 1]);
    return dp[0][n - 1];
    }
}

// 1689. Partitioning Into Minimum Number Of Deci-Binary Numbers
/* Just return max digit

Prove
Assume max digit in n is x.
Because deci-binary only contains 0 and 1,
we need at least x numbers to sum up a digit x.

Now we contruct an answer,
Take n = 135 as an example,
we initilize 5 deci-binary number with lengh = 3,
a1 = 000
a2 = 000
a3 = 000
a4 = 000
a5 = 000

For the first digit, we fill the first n[0] number with 1
For the second digit, we fill the first n[1] number with 1
For the third digit, we fill the first n[2] number with 1

So we have
a1 = 111
a2 = 011
a3 = 011
a4 = 001
a5 = 001

Finally, we have 111+11+11+1+1=135.


Complexity
Time O(L)
Space O(1)*/
public class Solution {
    public int MinPartitions(string n) {
        return n.Max(n => n -'0');  

    }
}
/*Solution: Return the max digit
Proof: For a given string, we find the maximum number m, 
we create m binary strings.
for each one, check each digit, if it’s greater than 0, 
we mark 1 at that position and decrease the digit by 1.

e.g. 21534
max is 5, we need five binary strings.
1. 11111: 21534 -> 10423
2. 10111: 10423 -> 00312
3: 00111: 00312 -> 00201
4: 00101: 00201 -> 00100
5: 00100: 00100 -> 00000

We can ignore the leading zeros.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MinPartitions(string n) {
        //return *max_element(begin(n), end(n)) - '0';
        int res = 0;
        for (int i = 0; i < n.Length; ++i)
            res = Math.Max(res, n[i] - '0');
        return res;
    }
}
public class Solution {
    public int MinPartitions(string n)  => n.Max(n => n -'0');  
}

public class Solution {
    public int MinPartitions(string n) {
        //return *max_element(begin(n), end(n)) - '0';
        int res = 0;
        for (int i = 0; i < n.Length; ++i)
            res = Math.Max(res,(int)Char.GetNumericValue(n[i]));
        return res;
    }
}

// 1688. Count of Matches in Tournament
/*Solution: Simulation / Recursion
Time complexity: O(logn)
Space complexity: O(1)*/
public class Solution {
    public int NumberOfMatches(int n) {
         int ans = 0;
    while (n > 1) {
      ans += n / 2 + (n & 1);
      n /= 2;
    }
    return ans;
    }
}

public class Solution {
    public int NumberOfMatches(int n) {
        return n > 1 ? n / 2 + (n & 1) + NumberOfMatches(n / 2) : 0;
    }
}

// 1687. Delivering Boxes from Storage to Ports
/*Solution: Sliding Window
Time complexity: O(n)
Space complexity: O(n)

*/
public class Solution {
    public int BoxDelivering(int[][] boxes, int portsCount, int maxBoxes, int maxWeight) {
        int n = boxes.Length;
    int[] dp = new int[n + 1]; // dp[i] := min trips to deliver boxes[0:i]
    for (int i = 0, j = 0, b = 0, w = 0, t = 1; j < n; ++j) {
      // Different ports.
      if (j == 0 || boxes[j][0] != boxes[j - 1][0]) ++t;      
      // Load the box.
      w += boxes[j][1];
      while (w > maxWeight  // Too heavy.
             || (j - i + 1) > maxBoxes // Too many boxes.
             || (i < j && dp[i + 1] == dp[i])) { // Same cost more boxes.
        w -= boxes[i++][1]; // 'Unload' the box.
        if (boxes[i][0] != boxes[i - 1][0]) --t; // Different ports.
      }
      // Takes t trips to deliver boxes[i~j].
      dp[j + 1] = dp[i] + t;
    }
    return dp[n];
    }
}

// 1686. Stone Game VI
/*Solution: Greedy
Sort by the sum of stone values.

Time complexity: O(nlogn)
Space complexity: O(n)

*/
public class Solution {
    public int StoneGameVI(int[] aliceValues, int[] bobValues) {
     int n = aliceValues.Length;
   List<KeyValuePair<int, int>> s = new List<KeyValuePair<int, int>>(n);    
    for (int i = 0; i < n; ++i)
      s.Add(new KeyValuePair<int, int>(aliceValues[i] + bobValues[i], i));
    s.Sort((a,b) => b.Key - a.Key);
    int ans = 0;
    for (int i = 0; i < n; ++i) {
      int idx = s[i].Value;
     
      ans += ((i & 1)> 0 ? bobValues[idx] : aliceValues[idx]) * ((i & 1)> 0 ? -1 : 1);
    }
    return (ans < 0) ? -1 : Convert.ToInt32(ans > 0);
    }
}

// 1679. Max Number of K-Sum Pairs
public class Solution {
    public int MaxOperations(int[] nums, int k) {
        Array.Sort(nums);
    int i = 0, j = nums.Length - 1;
    int ans = 0;
    while (i < j) {
      int s = nums[i] + nums[j];
      if (s == k) {
        ++ans;
        ++i; --j;
      } else if (s < k) {
        ++i;
      } else {
        --j;
      }
    }
    return ans;
    }
}

// 1680. Concatenation of Consecutive Binary Numbers
public class Solution {
    public int ConcatenatedBinary(int n) {
        int kMod = (int)1e9 + 7;
    long ans = 0;    
    for (int i = 1, len = 0; i <= n; ++i) {
      if ((i & (i - 1)) == 0) ++len;
      ans = ((ans << len) % kMod + i) % kMod;
    }
    return Convert.ToInt32(ans);
    }
}

// 1681. Minimum Incompatibility

// 1684. Count the Number of Consistent Strings
/*Solution: Hashtable
Time complexity: O(sum(len(word))
Space complexity: O(1)

*/
public class Solution {
    public int CountConsistentStrings(string allowed, string[] words) {
         int[] m = new int[26];
    foreach (char c in allowed) m[c - 'a'] = 1;
    int ans = words.Length;
    foreach (string w in words)  
        foreach (char c in w) 
            if (m[c - 'a'] == 0) {ans-= 1; break;}
            //ans += all_of(begin(w), end(w), [&m](char c) { return m[c - 'a']; });
    return ans;
    }
}

// 1685. Sum of Absolute Differences in a Sorted Array
/*Solution: Prefix Sum
Let s[i] denote sum(num[i] – num[j]) 0 <= j <= i
s[i] = s[i – 1] + (num[i] – num[i – 1]) * i
Let l[i] denote sum(nums[j] – nums[i]) i <= j < n
l[i] = l[i + 1] + (nums[i + 1] – num[i]) * (n – i – 1)
ans[i] = s[i] + l[i]

e.g. 1, 3, 7, 9
s[0] = 0
s[1] = 0 + (3 – 1) * 1 = 2
s[2] = 2 + (7 – 3) * 2 = 10
s[3] = 10 + (9 – 7) * 3 = 16
l[3] = 0
l[2] = 0 + (9 – 7) * 1 = 2
l[1] = 2 + (7 – 3) * 2 = 10
l[0] = 10 + (3 – 1) * 3 = 16

ans = [16, 12, 12, 16]

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int[] GetSumAbsoluteDifferences(int[] nums) {
          int n = nums.Length;
    int[] ans = new int[n];    
    for (int i = 1, sum = 0; i < n; ++i)
      ans[i] += (sum += (nums[i] - nums[i - 1]) * i);
    for (int i = n - 2, sum = 0; i >= 0; --i)
      ans[i] += (sum += (nums[i + 1] - nums[i]) * (n - i - 1));
    return ans;
    }
}

// 1675. Minimize Deviation in Array
/*Solution 2: Use Tree
Since odd can only be doubled once,
and even can not be doubled.
We can double all odds first.

(a & -a) returns the lowest bit of a (even number)
a / (a & -a) gives the smallest odd number that can divide a. */
public class Solution {
    public int MinimumDeviation(int[] nums) {
       PriorityQueue<int,int> pq = new PriorityQueue<int,int>();
        int n = nums.Length, mi = Int32.MaxValue, res = Int32.MaxValue;
        foreach (int a in nums) {
            int temp = a;
            if (temp % 2 == 1) temp *= 2;
            pq.Enqueue(-temp, -temp);
            mi = Math.Min(mi, temp);
        }
        while (true) {
            int a = -pq.Dequeue();
            res = Math.Min(res, a - mi);
            if (a % 2 == 1) break;
            mi = Math.Min(mi, a / 2);
            pq.Enqueue((-a / 2), (-a / 2));
        }
        return res;
      
    }
}
/*Solution: Priority Queue


If we double an odd number it becomes an even number, 
then we can only divide it by two which gives us back the original number. 
So we can pre-double all the odd numbers and only do division 
in the following process.

We push all numbers including pre-doubled odd ones onto a priority queue, 
and track the difference between the largest and smallest number.

Each time, we pop the largest number out and divide it by two 
then put it back to the priority queue, until the largest number becomes odd. 
We can not discard it and divide any other smaller numbers by two 
will only increase the max difference, so we can stop here.

ex1: [3, 5, 8] => [6, 8, 10] (pre-double) 
=> [5, 6, 8] => [4, 5, 6] => [3, 4, 5] max diff is 5 – 3 = 2
ex2: [4,1,5,20,3] => [2, 4, 6, 10, 20] (pre-double) 
=> [2, 4, 6, 10] => [2, 4, 5, 6] => [2,3,4,5] max diff = 5-2 = 3

Time complexity: O(n*logm*logn)
Space complexity: O(n)

C# use sortedset*/
public class Solution {
    public int MinimumDeviation(int[] nums) {
       SortedSet<int> s = new SortedSet<int>(nums.Select(x => (1 == (x & 1)) ? x * 2 : x));
   // foreach (int x in nums)
     // s.Add((x & 1) > 0 ? x * 2 : x);
    int ans = s.Max - s.Min; //SortedSet can use .Max and .Min
        
    while (s.Max % 2 == 0) {
      s.Add(s.Max / 2);
      s.Remove(s.Max);
      ans = Math.Min(ans, s.Max - s.Min);
    }
    return ans;
      
    }
}

// bit slower
public class Solution {
    public int MinimumDeviation(int[] nums) {
       SortedSet<int> s = new SortedSet<int>();
    foreach (int x in nums)
      s.Add((x & 1) > 0 ? x * 2 : x);
    int ans = s.Max - s.Min;
        
    while (s.Max % 2 == 0) {
      s.Add(s.Max / 2);
      s.Remove(s.Max);
      ans = Math.Min(ans, s.Max - s.Min);
    }
    return ans;
      
    }
}

// 1674. Minimum Moves to Make Array Complementary
/*Solution: Sweep Line / Prefix Sum
Let a = min(nums[i], nums[n-i-1]), b = max(nums[i], nums[n-i-1])
The key to this problem is how many moves do we need to make a + b == T.

if 2 <= T < a + 1, two moves, lower both a and b.
if a +1 <= T < a + b, one move, lower b
if a + b == T, zero move
if a + b + 1 <= T < b + limit + 1, one move, increase a
if b + limit + 1 <= T <= 2*limit, two moves, increase both a and b.

Time complexity: O(n + limit) or O(nlogn) if limit >>n
Space complexity: O(limit) or O(n)*/
public class Solution {
    public int MinMoves(int[] nums, int limit) {
        int n = nums.Length;
    int[] delta = new int[limit * 2 + 2];
    for (int i = 0; i < n / 2; ++i) {
      int a = Math.Min(nums[i], nums[n - i - 1]);
      int b = Math.Max(nums[i], nums[n - i - 1]);
      delta[2] += 2;          // dec a, dec b
      --delta[a + 1];         // dec a 
      --delta[a + b];         // no op
      ++delta[a + b + 1];     // inc a
      ++delta[b + limit + 1]; // inc a, inc b     
    }
    int ans = n;
    for (int t = 2, cur = 0; t <= limit * 2; ++t) {
      cur += delta[t];
      ans = Math.Min(ans, cur);
    }
    return ans;
    }
}

// 1673. Find the Most Competitive Subsequence
/*Solution: Stack
Use a stack to track the best solution so far, 
pop if the current number is less than the top of the stack and 
there are sufficient numbers left. 
Then push the current number to the stack if not full.

Time complexity: O(n)
Space complexity: O(k)*/
public class Solution {
    public int[] MostCompetitive(int[] nums, int k) {
        int n = nums.Length;
    int[] ans = new int[k];
    int c = 0;
    for (int i = 0; i < n; ++i) {
      while (c > 0 && ans[c - 1] > nums[i] && c + n - i - 1 >= k)
        --c;
      if (c < k) ans[c++] = nums[i];
    }
    return ans;
    }
}

// 1671. Minimum Number of Removals to Make Mountain Array
/*Solution: DP / LIS
LIS[i] := longest increasing subsequence ends with nums[i]
LDS[i] := longest decreasing subsequence starts with nums[i]
Let nums[i] be the peak, the length of the mountain array is LIS[i] + LDS[i] – 1
Note: LIS[i] and LDS[i] must be > 1 to form a valid mountain array.
ans = min(n – (LIS[i] + LDS[i] – 1)) 0 <= i < n

Time complexity: O(n^2)
Space complexity: O(n)

*/
public class Solution {
    public int MinimumMountainRemovals(int[] nums) {
         int n = nums.Length;
    int[] LIS = new int[n]; Array.Fill(LIS, 1); // LIS[i] := Longest increasing subseq ends with nums[i]
    int[] LDS = new int[n]; Array.Fill(LDS, 1); // LDS[i] := Longest decreasing subseq starts with nums[i]
    for (int i = 0; i < n; ++i)      
      for (int j = 0; j < i; ++j)
        if (nums[i] > nums[j]) LIS[i] = Math.Max(LIS[i], LIS[j] + 1);
    for (int i = n - 1; i >= 0; --i)
      for (int j = n - 1; j > i; --j)
        if (nums[i] > nums[j]) LDS[i] = Math.Max(LDS[i], LDS[j] + 1);
    int ans = Int32.MaxValue;
    for (int i = 0; i < n; ++i) {
      if (LIS[i] < 2 || LDS[i] < 2) continue;
      ans = Math.Min(ans, n - (LIS[i] + LDS[i] - 1));
    }
    return ans;
    }
}

// 1669. Merge In Between Linked Lists
/*Solution: List Operations
Find the following nodes:
1. previous node to the a-th node: prev_a
2. the b-th node: node_b
3. tail node of list2: tail2

prev_a->next = list2
tail2->next = node_b

return list1

Time complexity: O(m+n)
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
    public ListNode MergeInBetween(ListNode list1, int a, int b, ListNode list2) {
        ListNode dummy = new ListNode(0, list1);
    ListNode prev_a = dummy;
    for (int i = 0; i < a; ++i) prev_a = prev_a.next;
    ListNode node_b = prev_a.next;
    for (int i = a; i <= b; ++i) node_b = node_b.next;
    ListNode tail2 = list2;
    while (tail2.next != null) tail2 = tail2.next;
    
    prev_a.next = list2;    
    tail2.next = node_b;
    return list1;
    }
}

// 1664. Ways to Make a Fair Array
/*Solution: Prefix Sum
Time complexity: O(n)
Space complexity: O(n)

*/
public class Solution {
    public int WaysToMakeFair(int[] nums) {
        int n = nums.Length;
    int[] odds = new int[n + 1];
    int[] evens = new int[n + 1];
    for (int i = 0; i < n; ++i) {      
      odds[i + 1] = odds[i] + Convert.ToInt32(i % 2 == 1) * nums[i];
      evens[i + 1] = evens[i] + Convert.ToInt32(i % 2 == 0) * nums[i];      
    }
    int ans = 0;
    for (int i = 0; i < n; ++i) {
       int odd = odds[i] + (evens[n] - evens[i + 1]);
       int even = evens[i] + (odds[n] - odds[i + 1]);
      ans += Convert.ToInt32(odd == even);
    }
    return ans;
    }
}
// 1663. Smallest String With A Given Numeric Value
public class Solution {
    public string GetSmallestString(int n, int k) {
        // string ans(n, 'a');
         char[] ans = new char[n];
        Array.Fill(ans, 'a');
    k -= n;
    while (k > 0) {
      int d = Math.Min(k, 25);
     ans[--n] +=  (char)(d);
      //  ans[--n] = Convert.ToChar(ans[n] + d);
      k -= d;
    }
    return new string(ans);       
    }
}

public class Solution {
    public string GetSmallestString(int n, int k) {
        // string ans(n, 'a');
         char[] ans = new char[n];
        Array.Fill(ans, 'a');
    k -= n;
    while (k > 0) {
      int d = Math.Min(k, 25);
     // ans[--n] +=  (char)(d  'a');
        ans[--n] = Convert.ToChar(ans[n] + d);
      k -= d;
    }
    return new string(ans);
    }
}
// 1662. Check If Two String Arrays are Equivalent
/*
Solution1: Construct the string
Time complexity: O(l1 + l2)
Space complexity: O(l1 + l2)

Solution 2: Pointers
Time complexity: O(l1 + l2)
Space complexity: O(1)

*/
public class Solution {
    public bool ArrayStringsAreEqual(string[] word1, string[] word2) {
        int i1 = 0, j1 = 0;
    int i2 = 0, j2 = 0;
    while (i1 < word1.Length || i2 < word2.Length) {
      char c1 = i1 < word1.Length ? word1[i1][j1++] : '\0';
      char c2 = i2 < word2.Length ? word2[i2][j2++] : '\0';
      if (c1 != c2) return false;
      if (i1 < word1.Length && j1 == word1[i1].Length){
          ++i1; j1 = 0;
      }
        
      if (i2 < word2.Length && j2 == word2[i2].Length){
           ++i2; j2 = 0;
      }
       
    }
    return true;
    }
}

// 1658. Minimum Operations to Reduce X to Zero
/*Solution2: Sliding Window
Find the longest sliding window whose sum of elements equals sum(nums) – x
ans = n – window_size

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MinOperations(int[] nums, int x) {
        int n = nums.Length;
    int target = nums.Sum() - x;//accumulate(begin(nums), end(nums), 0) - x;    
    int ans = Int32.MaxValue;
    for (int s = 0, l = 0, r = 0; r < n; ++r) {
      s += nums[r];
      while (s > target && l <= r) s -= nums[l++];
      if (s == target) ans = Math.Min(ans, n - (r - l + 1));
    }
    return ans > n ? -1 : ans;
    }
}

// 1657. Determine if Two Strings Are Close
/*Solution: Hashtable
Two strings are close:
1. Have the same length, ccabbb => 6 == aabccc => 6
2. Have the same char set, ccabbb => (a, b, c) == aabccc => (a, b, c)
3. Have the same sorted char counts ccabbb => (1, 2, 3) == aabccc => (1, 2, 3)

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public bool CloseStrings(string word1, string word2) {
        int l1 = word1.Length;
     int l2 = word2.Length;
    if (l1 != l2) return false;
    int[] f1 = new int[128]; int[] f2 = new int[128];
         HashSet<char> s1 = new HashSet<char>();
        HashSet<char> s2 = new HashSet<char>();
    //int[] s1 = new int[128]; int[] s2 = new int[128];
    foreach (char c in word1) {++f1[c]; s1.Add(c);}//s1[c] = 1;}
    foreach (char c in word2) {++f2[c]; s2.Add(c);}//s2[c] = 1;}

   Array.Sort(f1);
   Array.Sort(f2);
    return f1.SequenceEqual(f2) && s1.SetEquals(s2); //SetEqual check for same Set
    }
}

public class Solution {
    public bool CloseStrings(string word1, string word2) {
        int l1 = word1.Length;
     int l2 = word2.Length;
    if (l1 != l2) return false;
    int[] f1 = new int[128]; int[] f2 = new int[128];
    int[] s1 = new int[128]; int[] s2 = new int[128];
    foreach (char c in word1) {++f1[c]; s1[c] = 1;}
    foreach (char c in word2) {++f2[c]; s2[c] = 1;}
   Array.Sort(f1);
   Array.Sort(f2);
    return f1.SequenceEqual(f2) && s1.SequenceEqual(s2); //SequenceEqual check for same Array
        
   
    }
}
//Linq
public class Solution {
    public bool CloseStrings(string word1, string word2) {
      // Words are close if:
        //  1. They consist of the same characters 
        if (word1.Except(word2).Any() || word2.Except(word1).Any())
            return false;
        
        // 2. Frequencies are equal 
        return word1
            .GroupBy(c => c)
            .Select(g => g.Count())
            .OrderBy(c => c)
            .SequenceEqual(word2
                .GroupBy(c => c)
                .Select(g => g.Count())
                .OrderBy(c => c));
   
    }
}

// 1656. Design an Ordered Stream
/*Solution: Straight Forward
Time complexity: O(n) in total
Space complexity: O(n)

*/
public class OrderedStream {

    public OrderedStream(int n) {
        data_ = new string[n + 1];
    }
    
    public IList<string> Insert(int idKey, string value) {
        data_[idKey] = value;
    IList<string> ans = new List<string>();
    if (ptr_ == idKey)
      while (ptr_ < data_.Length && data_[ptr_] != null)
        ans.Add(data_[ptr_++]);
    return ans;
    }
    

private int ptr_ = 1;
  private string[] data_ ;
}

// 1653. Minimum Deletions to Make String Balanced
/*Solution: DP
dp[i][0] := min # of dels to make s[0:i] all ‘a’s;
dp[i][1] := min # of dels to make s[0:i] ends with 0 or mode ‘b’s

if s[i-1] == ‘a’:
dp[i + 1][0] = dp[i][0], dp[i + 1][1] = min(dp[i + 1][0], dp[i][1] + 1)
else:
dp[i + 1][0] = dp[i][0] + 1, dp[i + 1][1] = dp[i][1]

Time complexity: O(n)
Space complexity: O(n) -> O(1)*/
public class Solution {
    public int MinimumDeletions(string s) {
    // const int n = s.length();
    // dp[i][0] := min # of dels to make s[0:i] all 'a's;
    // dp[i][1] := min # of dels to make s[0:i] ends with 0+ 'b's
    // vector<vector<int>> dp(n + 1, vector<int>(2));
    // for (int i = 0; i < n; ++i) {
    //   if (s[i] == 'a') {
    //     dp[i + 1][0] = dp[i][0];
    //     dp[i + 1][1] = min(dp[i + 1][0], dp[i][1] + 1);
    //   } else {
    //     dp[i + 1][0] = dp[i][0] + 1;
    //     dp[i + 1][1] = dp[i][1];
    //   }
    // }
    // return min(dp[n][0], dp[n][1]);
    int a = 0, b = 0;
    foreach (char c in s) {
      if (c == 'a') b = Math.Min(a, b + 1); 
      else ++a;
    }
    return Math.Min(a, b);
    }
}

// 1652. Defuse the Bomb
/*Solution 1: Simulation
Time complexity: O(n*k)
Space complexity: O(n)

*/
public class Solution {
    public int[] Decrypt(int[] code, int k) {
        int n = code.Length;
    int[] ans = new int[n];
    int sign = k > 0 ? 1 : -1;
    for (int i = 0; i < n; ++i)
      for (int j = 1; j <= Math.Abs(k); ++j)
        ans[i] += code[(i + j * sign + n) % n];
    return ans;
    }
}

// 1647. Minimum Deletions to Make Character Frequencies Unique
/*Solution: Hashtable
The deletion order doesn’t matter, we can process from ‘a’ to ‘z’. 
Use a hashtable to store the “final frequency” so far, 
for each char, decrease its frequency until it becomes unique 
in the final frequency hashtable.

Time complexity: O(n + 26^2)
Space complexity: O(26)*/
public class Solution {
    public int MinDeletions(string s) {
        int[] freq = new int[26];
    foreach (char c in s) ++freq[c - 'a'];    
    HashSet<int> seen = new HashSet<int>();
    int ans = 0;
    for (int i = 0;i< 26; i++) {
      while (freq[i] > 0 && !seen.Add(freq[i])) {        
        --freq[i];
        ++ans;
      }
    }
    return ans;
    }
}

// 1646. Get Maximum in Generated Array
/*Solution: Simulation
Generate the array by the given rules.

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int GetMaximumGenerated(int n) {
         int[] nums = new int[n + 1];
    nums[0] = 0;
    if (n > 0) nums[1] = 1;
    for (int i = 1; i * 2 <= n; ++i) {
      nums[2 * i] = nums[i];
      if (i * 2 + 1 <= n) nums[2 * i + 1] = nums[i] + nums[i + 1];
    }
    return nums.Max();//*max_element(begin(nums), end(nums));
    }
}
// 1641. Count Sorted Vowel Strings
/*Solution: DP
dp[i][j] := # of strings of length i ends with j.

dp[i][j] = sum(dp[i – 1][[k]) 0 <= k <= j

Time complexity: O(n)
Space complexity: O(n) -> O(1)

*/
public class Solution {
    public int CountVowelStrings(int n) {
        // dp([i])[j] = # of strings of length i ends with j
    int[] dp = new int[5];Array.Fill(dp, 1);
    for (int i = 2; i <= n; ++i)
      for (int j = 4; j >= 0; --j)        
        for (int k = 0; k < j; ++k)
          dp[j] += dp[k];
    return dp.Sum();//accumulate(begin(dp), end(dp), 0);
    }
}

// 1639. Number of Ways to Form a Target String Given a Dictionary
/*Solution: DP
dp[i][j] := # of ways to form target[0~j] 
where the j-th letter is from the i-th column of words.
count[i][j] := # of words that have word[i] == target[j]

dp[i][j] = dp[i-1][j-1] * count[i][j]

Time complexity: O(mn)
Space complexity: O(mn) -> O(n)*/
public class Solution {
    public int NumWays(string[] words, string target) {
         int kMod = (int)1e9 + 7;
  int n = target.Length;
   int m = words[0].Length;
    
    long[] dp = new long[n]; // dp[j] = # of ways to form t[0~j]
    for (int i = 0; i < m; ++i) {
      int[] count = new int[26];
      foreach (string word in words)
        ++count[word[i] - 'a'];      
      for (int j = Math.Min(i, n - 1); j >= 0; --j)
        dp[j] = (dp[j] + (j > 0 ? dp[j - 1] : 1) * 
                    count[target[j] - 'a']) % kMod;
    }
    return Convert.ToInt32(dp[n - 1]);
    }
}

// 1638. Count Substrings That Differ by One Character
/*Solution 2: Continuous Matching
Start matching s[0] with t[j] and s[i] with t[0]

Time complexity: O(mn)
Space complexity: O(1)*/
public class Solution {
    public int CountSubstrings(string s, string t) {
          int m = s.Length;
     int n = t.Length;
    int ans = 0;
   void helper(int i, int j) {      
      for (int cur = 0, pre = 0; i < m && j < n; ++i, ++j) {
        ++cur;
        if (s[i] != t[j]) {
          pre = cur;
          cur = 0;
        }
        ans += pre;
      }
    };
    for (int i = 0; i < m; ++i) helper(i, 0);
    for (int j = 1; j < n; ++j) helper(0, j);
    return ans;
    }
}

// 1637. Widest Vertical Area Between Two Points Containing No Points
/*Solution: Sort by x coordinates
Time complexity: O(nlogn)
Space complexity: O(n)*/
public class Solution {
    public int MaxWidthOfVerticalArea(int[][] points) {
       // sort(begin(points), end(points), [](const auto& p1, const auto& p2) {
    //  return p1[0] != p2[0] ? p1[0] < p2[0] : p1[1] < p2[1];
    //});
        Array.Sort(points, (a, b) => a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
    int ans = 0;
    for (int i = 1; i < points.Length; ++i)
      ans = Math.Max(ans, points[i][0] - points[i - 1][0]);
    return ans;
    }
}

// 1636. Sort Array by Increasing Frequency
/*Solution: Hashtable + Sorting
Use a hashtable to track the frequency of each number.

Time complexity: O(nlogn)
Space complexity: O(n)*/
public class Solution {
    public int[] FrequencySort(int[] nums) {
        Dictionary<int, int> freq = new Dictionary<int, int>();
    foreach (int x in nums) freq[x] = freq.GetValueOrDefault(x,0) + 1;
   /*sort(begin(nums), end(nums), [&](int a, int b) {
      if (freq[a] != freq[b]) return freq[a] < freq[b];
      return a > b;
    });*/
        Array.Sort(nums, (a, b) => freq[a] == freq[b] ? b - a: freq[a] - freq[b] );
    return nums;
    }
}

// 1629. Slowest Key
public class Solution {
    public char SlowestKey(int[] releaseTimes, string keysPressed) {
        int l = releaseTimes[0];
    char ans = keysPressed[0];
    
    for (int i = 1; i < releaseTimes.Length; ++i) {
      int t = releaseTimes[i] - releaseTimes[i - 1];
      if (t > l ) { 
        ans = keysPressed[i]; 
        l = t;
      } else if (t == l) {
        ans = Convert.ToChar('a' + Math.Max(ans - 'a', keysPressed[i] - 'a'));      
      }
    }
    return ans;
    }
}
/*Solution: Straightforward
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public char SlowestKey(int[] releaseTimes, string keysPressed) {
        int l = releaseTimes[0];
    char ans = keysPressed[0];
    
    for (int i = 1; i < releaseTimes.Length; ++i) {
      int t = releaseTimes[i] - releaseTimes[i - 1];
      if (t > l || (t == l &&  keysPressed[i] > ans)) { 
        ans = keysPressed[i]; 
        l = t;
      } 
    }
    return ans;
    }
}

// 1627. Graph Connectivity With Threshold
/*Solution: Union Find
For x, merge 2x, 3x, 4x, ..,
If a number is already “merged”, skip it.

Time complexity: O(nlogn? + queries)?
Space complexity: O(n)*/
public class Solution {
    public IList<bool> AreConnected(int n, int threshold, int[][] queries) {
        if (threshold == 0) return Enumerable.Repeat(true, queries.Length).ToList();
    
    int[] ds = Enumerable.Range(0, n+1).ToArray();//new int[n + 1];
    //iota(begin(ds), end(ds), 0);
    int find(int x) {
      return ds[x] == x ? x : ds[x] = find(ds[x]);
    };
    
    for (int x = threshold + 1; x <= n; ++x)
      if (ds[x] == x)
        for (int y = 2 * x; y <= n; y += x)    
          ds[Math.Max(find(x), find(y))] = Math.Min(find(x), find(y));
    
    IList<bool> ans = new List<bool>();
    foreach (int[] q in queries)
      ans.Add(find(q[0]) == find(q[1]));    
    return ans;
    }
}

// 1626. Best Team With No Conflicts
/*Solution: Sort + DP
Sort by (age, score) in descending order. For j < i, age[j] >= age[i]

dp[i] = max(dp[j] | score[j] >= score[i], j < i) + score[i]

Basically, we want to find the player j with best score among [0, i), 
and make sure score[i] <= score[j] 
(since age[j] >= age[i]) then we won’t have any conflicts.

ans = max(dp)*/
public class Solution {
    public int BestTeamScore(int[] scores, int[] ages) {
    int n = scores.Length;
    KeyValuePair<int, int>[] players = new KeyValuePair<int, int>[n];
    for (int i = 0; i < n; ++i) 
      players[i] = new KeyValuePair<int, int>(ages[i], scores[i]);
    Array.Sort(players, (a,b) => b.Key == a.Key? b.Value - a.Value : b.Key - a.Key);
    // dp[i] = max score of the first i players, i must be selected.
    int[] dp = new int[n];
    for (int i = 0; i < n; ++i) {      
      for (int j = 0; j < i; ++j)
        if (players[i].Value <= players[j].Value)
          dp[i] = Math.Max(dp[i], dp[j]);
      dp[i] += players[i].Value;
    }
    return dp.Max();//*max_element(begin(dp), end(dp));
    }
}

// 1625. Lexicographically Smallest String After Applying Operations
// Solution: Search
public class Solution {
    public string FindLexSmallestString(string s, int a, int b) {
        HashSet<String> seen = new HashSet<String>();    
    //string ans(s);
        String ans = new String(s);
    void dfs(string s) {
      if (!seen.Add(s)) return;
      //ans = (Math.Min(Convert.ToInt32(ans ), Convert.ToInt32(s))).ToString();
        if(string.Compare(s, ans) < 0)
			ans = s;
      StringBuilder t = new StringBuilder(s);
      for (int i = 1; i < s.Length; i += 2)
        t[i] = Convert.ToChar((t[i] - '0' + a) % 10 + '0');
      dfs(t.ToString());
      dfs(s.Substring(b) + s.Substring(0, b));
    };    
    dfs(s);
    return ans;
    }
}

// 1624. Largest Substring Between Two Equal Characters
/*Solution: Hashtable
Remember the first position each letter occurs.

Time complexity: O(n)
Space complexity: O(26)*/ 
public class Solution {
    public int MaxLengthBetweenEqualCharacters(string s) {
        int[] first = new int[26]; Array.Fill(first, -1);
    int ans = -1;
    for (int i = 0; i < s.Length; ++i) {
      int p = s[i] - 'a';
      if (first[p] != -1) {
        ans = Math.Max(ans, i - first[p] - 1);
      } else {
        first[p] = i;
      }      
    }
    return ans;
    }
}

// 1621. Number of Sets of K Non-Overlapping Line Segments
/*Solution 1: Naive DP (TLE)

dp[n][k] := ans of problem(n, k)
dp[n][1] = n * (n – 1) / 2 # C(n,2)
dp[n][k] = 1 if k == n – 1
dp[n][k] = 0 if k >= n
dp[n][k] = sum((i – 1) * dp(n – i + 1, k – 1) 2 <= i < n

Time complexity: O(n^2*k)
Space complexity: O(n*k)

Solution 2: DP w/ Prefix Sum

Time complexity: O(nk)
Space complexity: O(nk)

Solution 3: DP / 3D State

Time complexity: O(nk)
Space complexity: O(nk)

Solution 4: DP / Mathematical induction

Time complexity: O(nk)
Space complexity: O(nk)

Solution 5: DP / Reduction

This problem can be reduced to: given n + k – 1 points, 
pick k segments (2*k points).
if two consecutive points were selected by two segments 
e.g. i for A and i+1 for B, then they share a point in the original space.
Answer C(n + k – 1, 2*k)

Time complexity: O((n+k)*2) Pascal’s triangle
Space complexity: O((n+k)*2)*/
public class Solution {
    public int NumberOfSets(int n, int k) {
        int kMod = (int)1e9 + 7;
    int[][] dp = new int[n+k][];//(n + k , vector<int>(n + k));
    for (int i = 0; i < (n+k); ++i)
        dp[i] = new int[n+k]; 
    for (int i = 0; i < n + k; ++i) {
      dp[i][0] = dp[i][i] = 1;
      for (int j = 1; j < i; ++j)
        dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j]) % kMod;
    }
    return dp[n + k - 1][2 * k];
    }
}

// 1620. Coordinate With Maximum Network Quality
public class Solution {
    public int[] BestCoordinate(int[][] towers, int radius) {
        int n = 50;
    int[] ans = new int[2];
    int max_q = 0;    
    for (int x = 0; x <= n; ++x)
      for (int y = 0; y <= n; ++y) {
        int q = 0;
        foreach (var tower in towers) {
          int tx = tower[0], ty = tower[1];
          float d = (float)Math.Sqrt((x - tx) * (x - tx) + (y - ty) * (y - ty));
          if (d > radius) continue;
          q += (int)Math.Floor(tower[2] / (1 + d));
        }
        if (q > max_q) {
          max_q = q;
          ans = new int[2]{x, y};
        }
      }    
    return ans;
    }
}

// 1619. Mean of Array After Removing Some Elements
/*Solution: Sorting
Time complexity: O(nlogn)
Space complexity: O(1)*/
public class Solution {
    public double TrimMean(int[] arr) {
        Array.Sort(arr);
    int offset = arr.Length / 20;
        int sum = 0;
    
    // calc 1st m/2 + 1 element for 1st window
    for (int i = offset; i < arr.Length - offset; i++)
        sum += arr[i];
   // int sum = //accumulate(begin(arr) + offset, end(arr) - offset, 0);
    return (double)(sum) / (arr.Length - 2 * offset);
    }
}

// 1616. Split Two Strings to Make Palindrome
/*Solution: Greedy

Try to match the prefix of A and suffix of B (or the other way) 
as much as possible and then check 
whether the remaining part is a palindrome or not.

e.g. A = “abcxyzzz”, B = “uuuvvcba”
A’s prefix abc matches B’s suffix cba
We just need to check whether “xy” or “vv” is palindrome or not.
The concatenated string “abc|vvcba” is a palindrome, 
left abc is from A and vvcba is from B.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public bool CheckPalindromeFormation(string a, string b) {
         bool isPalindrome (string s, int i, int j) {
      while (i < j && s[i] == s[j]) {++i; --j;}        
      return i >= j;
    };
    bool check(string a, string b) {
      int i = 0;
      int j = a.Length - 1;
      while (i < j && a[i] == b[j]) {++i; --j;}        
      return isPalindrome(a, i, j) || isPalindrome(b, i, j);
    };
    return check(a, b) || check(b, a);
    }
}

public class Solution {
    public bool CheckPalindromeFormation(string a, string b) {
         bool isPalindrome (string s, int i, int j) {
      while (i < j && s[i] == s[j]) {++i; --j;}        
      return i >= j;
    };
    bool check(string a, string b) {
      for (int i = 0, j = a.Length - 1; i < j; ++i, --j)
            if (a[i] != b[j])
                return isPalindrome(a, i, j) || isPalindrome(b, i, j);
        return true;
      
    };
    return check(a, b) || check(b, a);
    }
}
/*Greedy Solution, O(1) Space

Explanation
Greedily take the a_suffix and b_prefix as long as they are palindrome,
that is, a_suffix = reversed(b_prefix).

The the middle part of a is s1,
The the middle part of b is s2.

If either s1 or s2 is palindrome, then return true.

Then we do the same thing for b_suffix and a_prefix


Solution 1:
Time O(N), Space O(N)*/
public class Solution {
    public bool CheckPalindromeFormation(string a, string b) {
         bool isPalindrome (string s, int i, int j) {
      for (; i < j; ++i, --j)
            if (s[i] != s[j])
                return false;
        return true;
    };
    bool check(string a, string b) {
      for (int i = 0, j = a.Length - 1; i < j; ++i, --j)
            if (a[i] != b[j])
                return isPalindrome(a, i, j) || isPalindrome(b, i, j);
        return true;
      
    };
    return check(a, b) || check(b, a);
    }
}

// 1614. Maximum Nesting Depth of the Parentheses
/*Solution: Stack
We only need to deal with ‘(‘ and ‘)’

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int MaxDepth(string s) {
        int ans = 0;
    int d = 0;
    foreach (char c in s) {
      if (c == '(') ans = Math.Max(ans, ++d);
      else if (c == ')') --d;
    }
    return ans;
    }
}

// 1611. Minimum One Bit Operations to Make Integers Zero
/*Solution 1: Graycode
Time complexity: O(logn)
Space complexity: O(1)

Ans is the order of n in graycode.*/
public class Solution {
    public int MinimumOneBitOperations(int n) {
         int ans = 0;
    while (n > 0) {
      ans ^= n;
      n >>= 1;
    }
    return ans;
    }
}

// 1610. Maximum Number of Visible Points
/*Solution: Sliding window

Sort all the points by angle, 
duplicate the points with angle + 2*PI to deal with turn around case.

maintain a window [l, r] such that angle[r] – angle[l] <= fov

Time complexity: O(nlogn)
Space complexity: O(n)

*/
public class Solution {
    public int VisiblePoints(IList<IList<int>> points, int angle, IList<int> location) {
        int at_origin = 0;
    List<double> ps = new List<double>();
    foreach (var p in points)
      if (p[0] == location[0] && p[1] == location[1])
        ++at_origin;
      else
        ps.Add(Math.Atan2(p[1] - location[1], p[0] - location[0]));
    ps.Sort();
    int n = ps.Count;
    for (int i = 0; i < n; ++i)
      ps.Add(ps[i] + 2.0 *  Math.PI); // duplicate the array +2PI
    int l = 0;
    int ans = 0;
    double fov = angle *  Math.PI / 180.0;
    for (int r = 0; r < ps.Count; ++r) {
      while (ps[r] - ps[l] > fov) ++l;
      ans = Math.Max(ans, r - l + 1);
    }
    return ans + at_origin;
        
    }
}

// 1609. Even Odd Tree
/*Solution 1: DFS
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
    public bool IsEvenOddTree(TreeNode root) {
        List<int> vals = new List<int>();
    bool dfs(TreeNode root, int d) {
      if (root == null) return true;
      if (vals.Count <= d)
        vals.Add(d % 2 == 0 ? -1 : Int32.MaxValue);
      int val = vals[d];
      if (d % 2 == 0)
        if (root.val % 2 == 0 || root.val <= val) return false;
      if (d % 2 == 1)
        if (root.val % 2 == 1 || root.val >= val) return false;
      vals[d] = root.val;
      return dfs(root.left, d + 1) && dfs(root.right, d + 1);
    };
    return dfs(root, 0);
    }
}

// 1600. Throne Inheritance
/*Solution: HashTable + DFS
Record :
1. mapping from parent to children (ordered)
2. who has dead

Time complexity: getInheritanceOrder O(n), other O(1)
Space complexity: O(n)*/
public class ThroneInheritance {

    public ThroneInheritance(string kingName) {
         king_ = kingName;
        m_[kingName] = m_.GetValueOrDefault( kingName,new List<string>());
    }
    
    public void Birth(string parentName, string childName) {
         m_[parentName] = m_.GetValueOrDefault( parentName,new List<string>());
         m_[parentName].Add(childName);
    }
    
    public void Death(string name) {
        dead_.Add(name);
    }
    
    public IList<string> GetInheritanceOrder() {
        IList<string> ans = new List<string>();
    void dfs(string name) {
      if (!dead_.Contains(name)) ans.Add(name);
        if (m_.ContainsKey(name)) {
             foreach (string child in m_[name]) dfs(child);
        }
     
    };
    dfs(king_);
    return ans;
    }
   
private string king_;  
  private Dictionary<string, List<string>> m_ = new Dictionary<string, List<string>>(); // parent -> list[children]
  private HashSet<string> dead_ = new HashSet<string>();
}

/**
 * Your ThroneInheritance object will be instantiated and called as such:
 * ThroneInheritance obj = new ThroneInheritance(kingName);
 * obj.Birth(parentName,childName);
 * obj.Death(name);
 * IList<string> param_3 = obj.GetInheritanceOrder();
 */

// 1601. Maximum Number of Achievable Transfer Requests
/*Solution: Combination
Try all combinations: O(2^n * (r + n))
Space complexity: O(n)*/
public class Solution {
    public int MaximumRequests(int n, int[][] requests) {
        int r = requests.Length; 
    int ans = 0;
    int[] nets = new int[n];
    for (int s = 0; s < (1 << r); ++s) {
     Array.Fill(nets, 0);// fill(begin(nets), end(nets), 0);
      for (int j = 0; j < r; ++j)
        if ((s & (1 << j)) > 0) {
          --nets[requests[j][0]];
          ++nets[requests[j][1]];
        }
         // check whether each building has 0 net requests
            bool valid = true;
            foreach (int k in nets){
                if (k != 0){
                    valid = false;
                    break;
                }
            }
            
            
      if (valid)
        ans = Math.Max(ans,BitOperations.PopCount((uint)s));
    }
    return ans;
    }
}

// 1604. Alert Using Same Key-Card Three or More Times in a One Hour Period
/*Solution: Hashtable + sorting
Time complexity: O(nlogn)
Space complexity: O(n)*/
public class Solution {
    public IList<string> AlertNames(string[] keyName, string[] keyTime) {
        Dictionary<string, List<int>> t = new Dictionary<string, List<int>>(); // {name -> time}
    for (int i = 0; i < keyName.Length; ++i) {
      int h = Convert.ToInt32(keyTime[i].Substring(0, 2));
      int m = Convert.ToInt32(keyTime[i].Substring(3));
         t[keyName[i]] = t.GetValueOrDefault(keyName[i], new List<int>());
      t[keyName[i]].Add(h * 60 + m);
    }
     List<string> ans = new List<string>();
    foreach (var (name, times) in t) {
      //sort(begin(times), end(times));
        times.Sort();
      for (int i = 2; i < times.Count; ++i)
        if (times[i] - times[i - 2] <= 60) {
          ans.Add(name);
          break;
        }   
    }
    //sort(begin(ans), end(ans));
        ans.Sort();
    return ans;
    }
}

// 1605. Find Valid Matrix Given Row and Column Sums
/*Solution: Greedy
Let a = min(row[i], col[j]), m[i][j] = a, row[i] -= a, col[j] -=a

Time complexity: O(m*n)
Space complexity: O(m*n)

*/
public class Solution {
    public int[][] RestoreMatrix(int[] rowSum, int[] colSum) {
        int m = rowSum.Length;
    int n = colSum.Length;
   // vector<vector<int>> ans(m, vector<int>(n));
        int[][] ans = new int[m][];
    for (int i = 0; i < m; ++i)
        ans[i] = new int[n]; 
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        ans[i][j] = Math.Min(rowSum[i], colSum[j]);
        rowSum[i] -= ans[i][j];
        colSum[j] -= ans[i][j];
      }
    return ans;
    }
}

// 1593. Split a String Into the Max Number of Unique Substrings
/*Solution: Brute Force
Try all combinations.
Time complexity: O(2^n)
Space complexity: O(n)*/
public class Solution {
    public int MaxUniqueSplit(string s) {
        int ans = 1;
    int n = s.Length;
    HashSet<string> seen = new HashSet<string>();
    void dfs(int p) {
      if (p == n) {        
        ans = Math.Max(ans, seen.Count);
        return;
      }
      for (int i = p; i < n; ++i) {
        string ss = s.Substring(p, i - p + 1);
        if (!seen.Add(ss)) continue;
        dfs(i + 1);
        seen.Remove(ss);
      }
    };
    dfs(0);
    return ans;
    }
}

// 1594. Maximum Non Negative Product in a Matrix
/*Solution: DP
Use two dp arrays,

dp_max[i][j] := max product of matrix[0~i][0~j]
dp_min[i][j] := min product of matrix[0~i][0~j]

Time complexity: O(m*n)
Space complexity: O(m*n)

*/
public class Solution {
    public int MaxProductPath(int[][] grid) {
        int kMod = (int)1e9 + 7;
    int m = grid.Length;
    int n = grid[0].Length;
   // vector<vector<long>> dp_max(m, vector<long>(n));
        long[][] dp_max = new long[m][];
    for (int i = 0; i < m; ++i)
       dp_max[i] = new long[n]; 
   // vector<vector<long>> dp_min(m, vector<long>(n));
         long[][] dp_min = new long[m][];
    for (int i = 0; i < m; ++i)
        dp_min[i] = new long[n]; 
    dp_max[0][0] = dp_min[0][0] = grid[0][0];
    for (int i = 1; i < m; ++i)
      dp_max[i][0] = dp_min[i][0] = dp_min[i - 1][0] * grid[i][0];
    for (int j = 1; j < n; ++j)
      dp_max[0][j] = dp_min[0][j] = dp_min[0][j - 1] * grid[0][j];
    for (int i = 1; i < m; ++i)
      for (int j = 1; j < n; ++j) {        
        if (grid[i][j] >= 0) {
          dp_max[i][j] = Math.Max(dp_max[i - 1][j], dp_max[i][j - 1]) * grid[i][j];
          dp_min[i][j] = Math.Min(dp_min[i - 1][j], dp_min[i][j - 1]) * grid[i][j];
        } else {
          dp_max[i][j] = Math.Min(dp_min[i - 1][j], dp_min[i][j - 1]) * grid[i][j];
          dp_min[i][j] = Math.Max(dp_max[i - 1][j], dp_max[i][j - 1]) * grid[i][j];
        }
      }
    return dp_max[m - 1][n - 1] >= 0 ? Convert.ToInt32(dp_max[m - 1][n - 1] % kMod) : -1;
    }
}

// 1595. Minimum Cost to Connect Two Groups of Points
/*Solution 1: Bistmask DP


dp[i][s] := min cost to connect first i (1-based) points in group1 and a set of points (represented by a bitmask s) in group2.

ans = dp[m][1 << n – 1]

dp[i][s | (1 << j)] := min(dp[i][s] + cost[i][j], dp[i-1][s] + cost[i][j])

Time complexity: O(m*n*2^n)
Space complexity: O(m*2^n)

Bottom-Up*/
public class Solution {
    public int ConnectTwoGroups(IList<IList<int>> cost) {
        int kInf = (int)1e9;
    int m = cost.Count;
    int n = cost[0].Count;
    // dp[i][s] := min cost to connect first i points in group1 
    // and points (bitmask s) in group2.
    //vector<vector<int>> dp(m + 1, vector<int>(1 << n, kInf));
    int[][] dp = new int[m+1][];
    for (int i = 0; i < (m+1); ++i){
        dp[i] = new int[(1 << n)]; Array.Fill(dp[i],kInf);
    }
    dp[0][0] = 0;
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        for (int s = 0; s < 1 << n; ++s)
          dp[i + 1][s | (1 << j)] = Math.Min(Math.Min(dp[i + 1][s | (1 << j)], 
                                         dp[i + 1][s] + cost[i][j]),
                                         dp[i][s] + cost[i][j]);
    return dp[m].Last();
    }
}

// 1598. Crawler Log Folder
/*Solution: Simulation
We only need to track the depth of current folder, 
and name and path can be ignored.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MinOperations(string[] logs) {
         int ans = 0;
    foreach (string log in logs) {
      if (log == "../")
        ans = Math.Max(ans - 1, 0);
      else if (log != "./")
        ++ans;
    }
    return ans;
    }
}

// 1599. Maximum Profit of Operating a Centennial Wheel
/*Solution: Simulation
Process if waiting customers > 0 or i < n.

Pruning, if runningCost > 4 * boardingCost (max revenue), 
there is no way to make profit.

Time complexity: sum(consumers) / 4
Space complexity: O(1)*/
public class Solution {
    public int MinOperationsMaxProfit(int[] customers, int boardingCost, int runningCost) {
          int n = customers.Length;
     int kMaxC = 4;
    if (runningCost > kMaxC * boardingCost) return -1;
    int c = 0;
    int ans = -1;
    int p = 0;
    int max_p = 0;    
    for (int r = 0; r < n || c > 0; ++r) {
      c += (r < n ? customers[r] : 0);      
      p += Math.Min(c, kMaxC) * boardingCost - runningCost;      
      c -= Math.Min(c, kMaxC);
      if (p > max_p) {
        max_p = p;
        ans = r + 1; // 1-based
      }
    }
    return ans;
    }
}

// 1589. Maximum Sum Obtained of Any Permutation
/*Solution: Greedy + Sweep line
Sort the numbers, and sort the frequency of each index, 
it’s easy to show largest number with largest frequency gives us max sum.

ans = sum(nums[i] * freq[i])

We can use sweep line to compute the frequency of each index in O(n) time and space.

For each request [start, end] : ++freq[start], –freq[end + 1]

Then the prefix sum of freq array is the frequency for each index.

Time complexity: O(nlogn)
Space complexity: O(n)

*/
public class Solution {
    public int MaxSumRangeQuery(int[] nums, int[][] requests) {
        int kMod = (int)1e9 + 7;
     int n = nums.Length;    
    long[] freq = new long[n];
    foreach (var r in requests) {
      ++freq[r[0]];
      if (r[1] + 1 < n) --freq[r[1] + 1];
    }
    for (int i = 1; i < n; ++i)
      freq[i] += freq[i - 1];
    
        Array.Sort(freq);
        Array.Sort(nums);
    //sort(begin(freq), end(freq));
    //sort(begin(nums), end(nums));
    
    long ans = 0;
    for (int i = 0; i < n; ++i)
      ans += freq[i] * nums[i];
    
    return Convert.ToInt32(ans % kMod);
    }
}

// 1590. Make Sum Divisible by P
/*Solution: HashTable + Prefix Sum
Very similar to subarray target sum.

Basically, we are trying to find a shortest subarray 
that has sum % p equals to r = sum(arr) % p.

We use a hashtable to store the last index of the prefix sum % p 
and check whether (prefix_sum + p – r) % p exists or not.

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int MinSubarray(int[] nums, int p) {
        int n = nums.Length;
        long sum =  0; Array.ForEach(nums, i => sum += i); int r = Convert.ToInt32(sum % p);
   // int r = nums.Sum() % p;//accumulate(begin(nums), end(nums), 0LL) % p;
    if (r == 0) return 0;
    Dictionary<int, int> m = new Dictionary<int, int>(){{0, -1}}; // {prefix_sum % p -> last_index}
    int s = 0;
    int ans = n;
    for (int i = 0; i < n; ++i) {
      s = (s + nums[i]) % p;
      //auto it = m.find((s + p - r) % p);
      //if (it != m.end())
    if (m.ContainsKey((s + p - r) % p))
        ans = Math.Min(ans, i - m[(s + p - r) % p]);
      m[s] = i;
    }
    return ans == n ? -1 : ans;
    }
}

// 1591. Strange Printer II
/*Solution: Dependency graph
For each color C find the maximum rectangle to cover it.
Any other color C’ in this rectangle is a dependency of C,
 e.g. C’ must be print first in order to print C.

Then this problem reduced to check if there is any cycle in the dependency graph.

e.g.
1 2 1
2 1 2
1 2 1
The maximum rectangle for 1 and 2 are both [0, 0] ~ [2, 2]. 
1 depends on 2, and 2 depends on 1. 
This is a circular reference and no way to print.

Time complexity: O(C*M*N)
Space complexity: O(C*C)*/
public class Solution {
    public bool IsPrintable(int[][] targetGrid) {
        int kMaxC = 60;
    int m = targetGrid.Length;
    int n = targetGrid[0].Length;
    List<HashSet<int>> deps = new List<HashSet<int>>(kMaxC + 1);
    for(int i = 0; i < (kMaxC +1); ++i) {
        deps.Add(new HashSet<int>());}
    for (int c = 1; c <= kMaxC; ++c) {
      int l = n, r = -1, t = m, b = -1;      
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
          if (targetGrid[i][j] == c){
               l = Math.Min(l, j); r = Math.Max(r, j); t = Math.Min(t, i); b = Math.Max(b, i);
          }
           
      if (l == -1) continue;
      for (int i = t; i <= b; ++i)
        for (int j = l; j <= r; ++j)
          if (targetGrid[i][j] > 0 && targetGrid[i][j] != c) 
               deps[c].Add(targetGrid[i][j]);
          
           // deps[c].Add(targetGrid[i][j]);
    }
    int[] seen = new int[kMaxC + 1];
    bool hasCycle(int c) {
      if (seen[c] == 1) return true;
      if (seen[c] == 2) return false;
      seen[c] = 1;
      foreach (int t in deps[c])
        if (hasCycle(t)) return true;
      seen[c] = 2;
      return false;
    };
    for (int c = 1; c <= kMaxC; ++c)
      if (hasCycle(c)) return false;
    return true;
    }
}

// 1592. Rearrange Spaces Between Words
/*Solution: Simulation
Time complexity: O(n)
Space complexity: O(n) -> O(1)*/
public class Solution {
    public string ReorderSpaces(string text) {
        List<string> words = new List<string>();
    string word = "";
    int t = 0;
    foreach (char c in text) {
      if (c == ' ') {
        ++t;
        if (word.Length != 0) {      
          words.Add(word);
          word = "";
        }
      } else {
        word += c;
      }
    }
    if (word.Length != 0) words.Add(word);
    int n = words.Count;    
    if (n == 1) return words[0] + new string(' ', t);
    int s = t / (n - 1);
    int r = t % (n - 1);
    StringBuilder ans = new ();
    for (int i = 0; i < words.Count; ++i) {
      if (i > 0) ans.Append(' ', s);
      ans.Append(words[i]);
    }
    ans.Append(' ', r);
    return ans.ToString();
    }
}

// 1579. Remove Max Number of Edges to Keep Graph Fully Traversable
/*Solution: Greedy + Spanning Tree / Union Find
Use type 3 (both) edges first.

Time complexity: O(E)
Space complexity: O(n)

*/
public class DSU {
public DSU(int n) {
    //iota(begin(p_), end(p_), 0);
    p_ = Enumerable.Range(0, n + 1).ToArray();
  }
  
  public int find(int x) {
    if (p_[x] == x) return x;
    return p_[x] = find(p_[x]);    
  }
  
  public int merge(int x, int y) {
    int rx = find(x);
    int ry = find(y);
    if (rx == ry) return 1;
    p_[rx] = ry;
    ++e_;
    return 0;
  }  
  
  public int edges() { return e_; }
private int[] p_;
 private int e_ = 0;
};
public class Solution {
    public int MaxNumEdgesToRemove(int n, int[][] edges) {
        int ans = 0;
    DSU A = new DSU(n);  DSU B = new DSU(n);
    foreach (var e in edges) {
      if (e[0] != 3) continue;
      ans += A.merge(e[1], e[2]);
      B.merge(e[1], e[2]);
    }
    foreach (var e in edges) {
      if (e[0] == 3) continue;
      DSU d = e[0] == 1 ? A : B;
      ans += d.merge(e[1], e[2]);
    }
    return (A.edges() == n - 1 && B.edges() == n - 1) ? ans : -1;
    }
}

// 1582. Special Positions in a Binary Matrix
/*Solution: Sum for each row and column
Brute force:
Time complexity: O(R*C*(R+C))
Space complexity: O(1)

We can pre-compute the sums for each row and each column, 
ans = sum(mat[r][c] == 1 and rsum[r] == 1 and csum[c] == 1)

Time complexity: O(R*C)
Space complexity: O(R+C)*/
public class Solution {
    public int NumSpecial(int[][] mat) {
         int rows = mat.Length;
    int cols = mat[0].Length;
    int[] rs = new int[rows];
    int[] cs = new int[cols];
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) {
        rs[r] += mat[r][c];
        cs[c] += mat[r][c];
      }
    int ans = 0;
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        ans += Convert.ToInt32(mat[r][c] > 0 && rs[r] == 1 && cs[c] == 1);      
    return ans;
    }
}

// 1583. Count Unhappy Friends
/*Solution: HashTable
Put the order in a map {x -> {y, order}}, 
since this is dense, we use can 2D array instead of hasthable which is much faster.

Then for each pair, we just need to check every other pair and compare their orders.

Time complexity: O(n^2)
Space complexity: O(n^2)

*/
public class Solution {
    public int UnhappyFriends(int n, int[][] preferences, int[][] pairs) {
        int[] p = new int[n];
    foreach (var pair in pairs) {
      p[pair[0]] = pair[1];
      p[pair[1]] = pair[0];
    }
    //vector<vector<int>> orders(n, vector<int>(n));
        int[][] orders = new int[n][];  
         for(int i = 0; i < n; i++){
             orders[i] = new int[n];
         }
    for (int x = 0; x < n; ++x)
      for (int i = 0; i < preferences[x].Length; ++i)
        orders[x][preferences[x][i]] = i;
    int ans = 0;
    for (int x = 0; x < n; ++x) {
      int y = p[x];      
      bool found = false;      
      for (int u = 0; u < n && !found; ++u) {
        if (u == x || u == y) continue;
        int v = p[u];
        found |= (orders[x][u] < orders[x][y] && orders[u][x] < orders[u][v]); 
      }
      if (found) ++ans;
    }
    return ans;
    }
}

// 1584. Min Cost to Connect All Points
/*Prim’s Algorithm
ds[i] := min distance from i to ANY nodes in the tree.

Time complexity: O(n^2) Space complexity: O(n)*/
public class Solution {
    public int MinCostConnectPoints(int[][] points) {
        int n = points.Length;
    int dist(int[] pi, int[] pj) {
      return Math.Abs(pi[0] - pj[0]) + Math.Abs(pi[1] - pj[1]);
    };
    int[] ds = new int[n];Array.Fill(ds, Int32.MaxValue);  
    for (int i = 1; i < n; ++i)
      ds[i] = dist(points[0], points[i]);
    
    int ans = 0;
    for (int i = 1; i < n; ++i) {
      //int it = ds.Min();//min_element(begin(ds), end(ds));
      int v = Array.IndexOf(ds, ds.Min()) ;//distance(begin(ds), it);
      ans += ds[v];      
      ds[v] =  Int32.MaxValue; // done
      for (int j = 0; j < n; ++j) {
        if (ds[j] ==  Int32.MaxValue) continue;
        ds[j] = Math.Min(ds[j], dist(points[j], points[v]));
      }        
    }
    return ans;
    }
}

// 1585. Check If String Is Transformable With Substring Sort Operations
/*Solution: Queue

We can move a smaller digit from right to left by sorting two adjacent digits.
e.g. 18572 -> 18527 -> 18257 -> 12857, 
but we can not move a larger to the left of a smaller one.

Thus, for each digit in the target string, 
we find the first occurrence of it in s, 
and try to move it to the front by checking if there is any smaller one in front of it.

Time complexity: O(n)
Space complexity: O(n)

*/
public class Solution {
    public bool IsTransformable(string s, string t) {
        LinkedList<int>[] idx = new LinkedList<int>[10];
    for (int i = 0; i < 10; ++i)
      idx[i] = new LinkedList<int>();
        
    for (int i = 0; i < s.Length; ++i)
    {
        idx[s[i] - '0'].AddLast(i);
    }
    foreach (char c in t) {
       int d  = c - '0';
      if (idx[d].Count == 0) return false;
      for (int i = 0; i < d; ++i){
           if (idx[d].Count != 0 && idx[i].Count != 0 && idx[i].First.Value < idx[d].First.Value){
             return false;
        }
      }
      idx[d].RemoveFirst();      
    }
    return true;
    }
}

// 1575. Count All Possible Routes
/*Solution: DP
dp[j][f] := # of ways to start from city ‘start’ to reach city ‘j’ with fuel level f.

dp[j][f] = sum(dp[i][f + d]) d = dist(i, j)

init: dp[start][fuel] = 1

Time complexity: O(n^2*fuel)
Space complexity: O(n*fuel)

*/
public class Solution {
    public int CountRoutes(int[] locations, int start, int finish, int fuel) {
        int kMod = (int)1e9 + 7;
     int n = locations.Length;    
   // vector<vector<int>> dp(n, vector<int>(fuel + 1));
        int[][] dp = new int[n][]; 
    for(int i = 0; i < n; i++) 
    {dp[i] = new int[fuel + 1];}
    dp[start][fuel] = 1;
    for (int f = fuel; f > 0; --f)
      for (int i = 0; i < n; ++i) {
        if (dp[i][f] == 0 || Math.Abs(locations[i] - locations[finish]) > f) continue; // pruning.
        for (int j = 0; j < n; ++j) {
          int d = Math.Abs(locations[i] - locations[j]);                    
          if (i == j || f < d) continue;
          dp[j][f - d] = (dp[j][f - d] + dp[i][f]) % kMod;
        }
      }
   // return accumulate(begin(dp[finish]), end(dp[finish]), 0LL) % kMod;
        long sum = 0; Array.ForEach(dp[finish], i => sum += i);
        return Convert.ToInt32( sum % kMod);
    }
}

// 1576. Replace All ?'s to Avoid Consecutive Repeating Characters
/*Solution: Greedy
For each ?, find the first one among ‘abc’ that is not same as left or right.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public string ModifyString(string s) {
        int n = s.Length;
        char[] arr = s.ToCharArray();
    for (int i = 0; i < n; ++i) {
      if (arr[i] != '?') continue;
      foreach (char c in "abc")
        if ((i == 0 || arr[i - 1] != c) && (i == n - 1 || arr[i + 1] != c)) {
          arr[i] = c;//s[i] = c;
          break;
        }
    }
    return new String(arr);
    }
}

// 1577. Number of Ways Where Square of Number Is Equal to Product of Two Numbers
/*Solution: Hashtable
For each number y in the second array, count its frequency.

For each number x in the first, if x * x % y == 0, let r = x * x / y
if r == y: ans += f[y] * f[y-1]
else ans += f[y] * f[r]

Final ans /= 2

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int NumTriplets(int[] nums1, int[] nums2) {
        return solve(nums1, nums2) + solve(nums2, nums1);
    }
    
  
private int solve(int[] nums1, int[] nums2) {
    int ans = 0;
    Dictionary<int, int> f = new Dictionary<int, int>();
    foreach (int y in nums2) f[y] = f.GetValueOrDefault(y,0)+1;
    foreach (long x in nums1)
      foreach (var (y, c) in f) {                
        long r = x * x / y;
        if ((x * x % y) > 0 || !f.ContainsKey((int)r)) continue;
        if (r == y) ans += c * (c - 1);
        else ans += c * f[(int)r];
      }
    return ans / 2;
}
    
}

// 1578. Minimum Time to Make Rope Colorful
/*Solution: Group by group
For a group of same letters, delete all expect the one with the highest cost.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MinCost(string colors, int[] neededTime) {
        int t = neededTime[0];
    int m = neededTime[0];
    int ans = 0;
    for (int i = 1; i < colors.Length; ++i) {
      if (colors[i] != colors[i - 1]) {
        ans += t - m;
        t = m = 0;
      }
      t += neededTime[i];
      m = Math.Max(m, neededTime[i]);
    }
    return ans + (t - m);
    }
}

// 1573. Number of Ways to Split a String
/*Solution: Counting

Count how many ones in the binary string as T, if not a factor of 3, 
then there is no answer.

Count how many positions that have prefix sum of T/3 as l, 
and how many positions that have prefix sum of T/3*2 as r.

Ans = l * r

But we need to special handle the all zero cases, 
which equals to C(n-2, 2) = (n – 1) * (n – 2) / 2

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int NumWays(string s) {
         int kMod = (int)1e9 + 7;
    long n = s.Length;
    int t = 0;
    foreach (char c in s) t += Convert.ToInt32(c == '1');
    if ((t % 3) > 0) return 0;
    if (t == 0)
      return Convert.ToInt32(((1 + (n - 2)) * (n - 2) / 2) % kMod);
    t /= 3;
    long l = 0;
    long r = 0;
    for (int i = 0, c = 0; i < n; ++i) {
      c += Convert.ToInt32(s[i] == '1');
      if (c == t) ++l;
      else if (c == t * 2) ++r;
    }
    return Convert.ToInt32((l * r) % kMod);
    }
}

// 1569. Number of Ways to Reorder Array to Get Same BST
/*Solution: Recursion + Combinatorics

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
Space complexity: O(n^2)*/
public class Solution {
    public int NumOfWays(int[] nums) {
         int n = nums.Length;
    int kMod = (int)1e9 + 7;
   // vector<vector<int>> cnk(n + 1, vector<int>(n + 1, 1));  
      int[][] cnk = new int[n + 1][];
      for (int a = 0; a < (n + 1); ++a){
        cnk[a] = new int[n + 1];Array.Fill(cnk[a],1);}
    for (int i = 1; i <= n; ++i)      
      for (int j = 1; j < i; ++j)
        cnk[i][j] = (cnk[i - 1][j] + cnk[i - 1][j - 1]) % kMod;    
    int trees(List<int> nums) {
      int n = nums.Count;
      if (n <= 2) return 1;
      List<int> left = new List<int>();
      List<int> right = new List<int>();
      for (int i = 1; i < nums.Count; ++i)
        if (nums[i] < nums[0]) left.Add(nums[i]);
        else right.Add(nums[i]);
      long ans = cnk[n - 1][left.Count];
      ans = (ans * trees(left)) % kMod;      
      ans = (ans * trees(right)) % kMod;      
      return (int)(ans);
    };
    return trees(nums.ToList()) - 1;
    }
}

// 1560. Most Visited Sector in a Circular Track
/*Solution: Simulation
Time complexity: O(m*n)
Space complexity: O(n)*/
public class Solution {
    public IList<int> MostVisited(int n, int[] rounds) {
        int[] counts = new int[n];
    counts[rounds[0] - 1] = 1;
    for (int i = 1; i < rounds.Length; ++i)
      for (int s = rounds[i - 1]; ; ++s) {
        ++counts[s %= n];
        if (s == rounds[i] - 1) break;
      }
    int max_count = counts.Max();//*max_element(begin(counts), end(counts));
     IList<int> ans = new List<int>();
    for (int i = 0; i < n; ++i)      
      if (counts[i] == max_count) ans.Add(i + 1);    
    return ans;
    }
}

// 1561. Maximum Number of Coins You Can Get
/*Solution: Greedy
Always take the second largest element of a in the sorted array.
[1, 2, 3, 4, 5, 6, 7, 8, 9]
tuples: (1, 8, 9), (2, 6, 7), (3, 4, 5)
Alice: 9, 7, 5
You: 8, 6, 4
Bob: 1, 2, 3

Time complexity: O(nlogn) -> O(n + k)
Space complexity: O(1)

Counting Sort*/
public class Solution {
    public int MaxCoins(int[] piles) {
        int kMax = 10000;
    int n = piles.Length / 3;
    int[] counts = new int[kMax + 1];
    foreach (int v in piles) ++counts[v];
    int idx = 0;
    for (int i = 1; i <= kMax; ++i)
      while (counts[i]-- > 0) piles[idx++] = i;
    int ans = 0;
    for (int i = 0; i < n; ++i)
      ans += piles[n * 3 - 2 - i * 2];
    return ans;
    }
}

// 1563. Stone Game V
/*Solution: Range DP + Prefix Sum


dp[l][r] := max store Alice can get from range [l, r]
sum_l = sum(l, k), sum_r = sum(k + 1, r)
dp[l][r] = max{
dp[l][k] + sum_l if sum_l < sum_r
dp[k+1][r] + sum_r if sum_r < sum_l
max(dp[l][k], dp[k+1][r])) + sum_l if sum_l == sum_r)
} for k in [l, r)

Time complexity: O(n^3)
Space complexity: O(n^2)*/
public class Solution {
    public int StoneGameV(int[] stoneValue) {
        int n = stoneValue.Length;
    int[] sums = new int[n + 1];//(n + 1);
    for (int i = 0; i < n; ++i)
      sums[i + 1] = sums[i] + stoneValue[i];
   // vector<vector<int>> cache(n, vector<int>(n, -1));
        int[][] cache = new int[n][];
   for (int a = 0; a < n; ++a){
        cache[a] = new int[n];
        Array.Fill(cache[a], -1);}
    // max value alice can get from range [l, r]
    int dp(int l, int r) {
      if (l == r) return 0;
      int ans = cache[l][r];
      if (ans != -1) return ans;
      for (int k = l; k < r; ++k) {
        // left: [l, k], right: [k + 1, r]
        int sum_l = sums[k + 1] - sums[l];
        int sum_r = sums[r + 1] - sums[k + 1];
        if (sum_l > sum_r)
          ans = Math.Max(ans, sum_r + dp(k + 1, r));
        else if (sum_l < sum_r)
          ans = Math.Max(ans, sum_l + dp(l, k));
        else
          ans = Math.Max(ans, sum_l + Math.Max(dp(l, k), dp(k + 1, r)));
      }    
        cache[l][r] = ans;
      return ans;
    };
    
    return dp(0, n - 1);
    }
}

// 1566. Detect Pattern of Length M Repeated K or More Times
/*Solution 2: Shift and count
Since we need k consecutive subarrays, we can compare arr[i] with arr[i + m], 
if they are the same, increase the counter, otherwise reset the counter. 
If the counter reaches (k – 1) * m, 
it means we found k consecutive subarrays of length m.

ex1: arr = [1,2,4,4,4,4], m = 1, k = 3
i arr[i], arr[i + m] counter
0 1. 2. 0
0 2. 4. 0
0 4. 4. 1
0 4. 4. 2. <– found

ex2: arr = [1,2,1,2,1,1,1,3], m = 2, k = 2
i arr[i], arr[i + m] counter
0 1. 1. 1
0 2. 2. 2 <– found

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public bool ContainsPattern(int[] arr, int m, int k) {
        int n = arr.Length;
    int count = 0;
    for (int i = 0; i + m < n; ++i) {      
      if (arr[i] == arr[i + m]) {
        if (++count == (k - 1) * m) return true;
      } else {
        count = 0;
      }      
    }
    return false;
    }
}

// 1553. Minimum Number of Days to Eat N Oranges
/*Solution 2: BFS

if x % 2 == 0, push x/2 onto the queue
if x % 3 == 0, push x/3 onto the queue
always push x – 1 onto the queue

*/
public class Solution {
    public int MinDays(int n) {
        Queue<int> q = new Queue<int>();q.Enqueue(n);
    HashSet<int> seen = new HashSet<int>();
    int steps = 0;
    while (q.Count != 0) {
      int size = q.Count;
      while ((size--) > 0) {
        int x = q.Peek(); q.Dequeue();        
        if (x == 0) return steps;
        if (x % 2 == 0 && seen.Add(x / 2))
          q.Enqueue(x / 2);                  
        if (x % 3 == 0 && seen.Add(x / 3))
          q.Enqueue(x / 3);        
        if (seen.Add(x - 1))
          q.Enqueue(x - 1);        
      }
      ++steps;
    }
    return -1;
  }
}
/*Solution: Greedy + DP

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
Time complexity: O(n) // w/o memoization, close to O(logn) in practice.
Space complexity: O(logn)*/
public class Solution {
    public int MinDays(int n) {
        return dp(n);
    }
    
    private static Dictionary<int, int> cache = new Dictionary<int, int>();
  
  private int dp(int n) {
    if (n <= 1) return n;
    if (cache.ContainsKey(n)) return cache[n];    
    int ans = 1 + Math.Min(n % 2 + dp(n / 2), n % 3 + dp(n / 3));
    cache.Add(n, ans);
    return ans;
  }
}

// 1556. Thousand Separator
/*Solution: Digit by digit

Time complexity: O(log^2(n)) -> O(logn)
Space complexity: O(log(n))*/
public class Solution {
    public string ThousandSeparator(int n) {
        StringBuilder ans = new StringBuilder();
    int count = 0;
    do {
      if (count++ % 3 == 0 && ans.Length > 0)
        ans.Insert(0, '.');   
      ans.Insert(0, n % 10);
      n /= 10;      
    } while (n > 0);
    return ans.ToString();
    }
}
//Linq
public class Solution {
    public string ThousandSeparator(int n) => String.Format("{0:#,##0}", n).Replace(',','.');
}

// 1558. Minimum Numbers of Function Calls to Make Target Array
/*Solution: count 1s


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
Space complexity: O(1)*/
public class Solution {
    public int MinOperations(int[] nums) {
        int ans = 0;
    int high = 0;
    for (int i = 0; i < nums.Length; i++) {
      int l = -1;
      while (nums[i] != 0) {        
        ans += nums[i] & 1;
        nums[i] >>= 1; 
        ++l;
      }
      high = Math.Max(high, l);
    }
    return ans + high;
    }
}

// 1559. Detect Cycles in 2D Grid
/*Solution: DFS
Finding a cycle in an undirected graph => visiting a node 
that has already been visited and it’s not the parent node of the current node.
b b
b b
null -> (0, 0) -> (0, 1) -> (1, 1) -> (1, 0) -> (0, 0)
The second time we visit (0, 0) which has already been visited before 
and it’s not the parent of the current node (1, 0) ( (1, 0)’s parent is (1, 1) ) 
which means we found a cycle.

Time complexity: O(m*n)
Space complexity: O(m*n)*/
public class Solution {
    public bool ContainsCycle(char[][] grid) {
         int m = grid.Length;
        int n = grid[0].Length;
    //vector<vector<int>> seen(m, vector<int>(n));
        int[][] seen = new int[m][];
   for (int a = 0; a < m; ++a){
        seen[a] = new int[n];}
    int[] dirs = new int[5]{0, 1, 0, -1, 0};
    bool dfs(int i, int j, int pi, int pj) {
      ++seen[i][j];
      for (int d = 0; d < 4; ++d) {
        int ni = i + dirs[d];
        int nj = j + dirs[d + 1];
        if (ni < 0 || nj < 0 || ni >= m || nj >= n) continue;
        if (grid[ni][nj] != grid[i][j]) continue;
        if (seen[ni][nj] == 0) {
          if (dfs(ni, nj, i, j)) return true;
        } else if (ni != pi || nj != pj) {
          return true;
        }       
      }
      return false;
    };    
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        if (seen[i][j]++ == 0 && dfs(i, j, -1, -1)) 
          return true;
    return false;
    }
}

// 1546. Maximum Number of Non-Overlapping Subarrays With Sum Equals Target
/*Solution: Prefix Sum + DP
Use a hashmap index to record the last index when a given prefix sum occurs.
dp[i] := max # of non-overlapping subarrays of nums[0~i], 
nums[i] is not required to be included.
dp[i+1] = max(dp[i], // skip nums[i]
dp[index[sum – target] + 1] + 1) // use nums[i] to form a new subarray
ans = dp[n]

Time complexity: O(n)
Space complexity: O(n)*/
 public class Solution {
    public int MaxNonOverlapping(int[] nums, int target) {
        int n = nums.Length;
    int[] dp= new int[n + 1];Array.Fill(dp, 0); // ans at nums[i];
    Dictionary<int, int> index = new Dictionary<int, int>(); // {prefix sum -> last_index}
    index[0] = -1;    
    int sum = 0;
    for (int i = 0; i < n; ++i) {
      sum += nums[i];
      int t = sum - target;
      dp[i + 1] = dp[i]; 
      if (index.ContainsKey(t))
        dp[i + 1] = Math.Max(dp[i + 1], dp[index[t] + 1] + 1);
      index[sum] = i;      
    }
    return dp[n];
    }
}

// 1547. Minimum Cost to Cut a Stick
/*Solution: Range DP
dp[i][j] := min cost to finish the i-th cuts to the j-th (in sorted order)
dp[i][j] = r – l + min(dp[i][k – 1], dp[k + 1][j]) 
# [l, r] is the current stick range.

Time complexity: O(n^3)
Space complexity: O(n^2)*/
public class Solution {
    public int MinCost(int n, int[] cuts) {
         Array.Sort(cuts);
    int c = cuts.Length;
    int[][] dp = new int[c][];
    for (int a = 0; a < c; ++a){
        dp[a] = new int[c];}
    return solve(dp, cuts, 0, c - 1, 0, n);
    }    

  
  private int solve(int[][] dp, int[] cuts, int i, int j, int l, int r) {
    if (i > j) return 0;
    if (i == j) return r - l;
    if (dp[i][j] != 0) return dp[i][j];
    int ans = Int32.MaxValue;
    for (int k = i; k <= j; ++k)
      ans = Math.Min(ans, r - l 
                          + solve(dp, cuts, i, k - 1, l, cuts[k])
                          + solve(dp, cuts, k + 1, j, cuts[k], r));
    return dp[i][j] = ans;
  }
}

// 1550. Three Consecutive Odds
/*Solution: Counting
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public bool ThreeConsecutiveOdds(int[] arr) {
        int count = 0;
    foreach(int x in arr) {
      if ((x & 1) > 0) {
        if (++count == 3) return true;
      } else {
        count = 0;
      }
    }
    return false;  
    }
}

// 1551. Minimum Operations to Make Array Equal
/*Solution: Math
1: Find the mean (final value) of the array, assuming x, easy to show x == n
2: Compute the sum of an arithmetic progression of (x – 1) + (x – 3) + … 
for n // 2 pairs

e.g. n = 6
arr = [1, 3, 5, 7, 9, 11]
x = (1 + 2 * n – 1) / 2 = 6 = n
steps = (6 – 1) + (6 – 3) + (6 – 5) = (n // 2) * (n – (1 + n – 1) / 2) 
= (n // 2) * (n – n // 2) = 3 * 3 = 9

e.g. n = 5
arr = [1,3,5,7,9]
x = (1 + 2 * n – 1) / 2 = 5 = n
steps = (5 – 1) + (5 – 3)= (n//2) * (n – n // 2) 
= (n // 2) * ((n + 1) // 2) = 2 * 3 = 6

Time complexity: O(1)
Space complexity: O(1)*/
public class Solution {
    public int MinOperations(int n) {
        return (n / 2) * ((n + 1) / 2);
    }
}

// 1545. Find Kth Bit in Nth Binary String
/*Solution 2: Recursion
All the strings have odd length of L = (1 << n) – 1,
Let say the center m = (L + 1) / 2
if n == 1, k should be 1 and ans is “0”.
Otherwise
if k == m, we know it’s “1”.
if k < m, the answer is the same as find(n-1, K)
if k > m, we are finding a flipped and mirror char in S(n-1), 
thus the answer is flip(find(n-1, L – k + 1)).

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public char FindKthBit(int n, int k) {
        if (n == 1) return '0';
    int l = (1 << n) - 1;
    if (k == (l + 1) / 2) return '1';
    else if (k < (l + 1) / 2) 
      return FindKthBit(n - 1, k);
    else
      return (char)('1' - FindKthBit(n - 1, l - k + 1) + '0');
    }
}

// 1544. Make The String Great
/*Solution: Stack
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
Space complexity: O(n)*/
public class Solution {
    public string MakeGood(string s) {
        var sb = new StringBuilder();
    for (int i = 0; i < s.Length; ++i) {
      int l = sb.Length;
      if (l > 0 && Math.Abs(sb[l - 1] - s[i]) == 32) {
        sb.Length = l - 1; // remove last char
      } else {
        sb.Append(s[i]);
      }
    }
    return sb.ToString();
    }
}

// 1542. Find Longest Awesome Substring
/*Solution: Prefix mask + Hashtable


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
Space complexity: O(2^10) = O(1)*/
public class Solution {
    public int LongestAwesome(string s) {
        int n = s.Length;
    int[] idx = new int[1024];
    Array.Fill(idx, n);
    idx[0] = -1;
    int mask = 0;
    int ans = 0;
    for (int i = 0; i < n; ++i) {
      mask ^= (1 << (s[i] - '0'));
      ans = Math.Max(ans, i - idx[mask]);      
      for (int j = 0; j < 10; ++j)
        ans = Math.Max(ans, 
                       i - idx[mask ^ (1 << j)]);
      idx[mask] = Math.Min(idx[mask], i);
    }
    return ans;
    }
}

// 1541. Minimum Insertions to Balance a Parentheses String
/*Solution: Counting
Count how many close parentheses we need.

if s[i] is ‘)’, we decrease the counter.
if counter becomes negative, means we need to insert ‘(‘
increase ans by 1, increase the counter by 2, we need one more ‘)’
‘)’ -> ‘()’
if s[i] is ‘(‘
if we have an odd counter, means there is a unbalanced ‘)’ e.g. ‘(()(‘, counter is 3
need to insert ‘)’, decrease counter, increase ans
‘(()(‘ -> ‘(())(‘, counter = 2
increase counter by 2, each ‘(‘ needs two ‘)’s. ‘(())(‘ -> counter = 4
Once done, if counter is greater than zero, we need insert that much ‘)s’
counter = 5, ‘((()’ -> ‘((())))))’
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MinInsertions(string s) {
         int ans = 0;
    int close = 0; // # of ')' needed.    
    foreach (char c in s) {
      if (c == ')') {
        if (--close < 0) {          
          // need to insert one '('
          // ')' -> '()'
          ++ans;
          close += 2;
        }
      } else {
        if ((close & 1) > 0) {          
          // need to insert one ')'
          // case '(()(' -> '(())('
          --close;
          ++ans;
        }
        close += 2; // need two ')'s
      }
    }
    return ans + close;
    }
}

// 1540. Can Convert String in K Moves
/*Solution: HashTable

Count how many times a d-shift has occurred.
a -> c is a 2-shift, z -> b is also 2-shift
a -> d is a 3-shift
a -> a is a 0-shift that we can skip
if a d-shift happened for the first time, we need at least d moves
However, if it happened for c times, we need at least d + 26 * c moves
e.g. we can do a 2-shift at the 2nd move, 
do another one at 2 + 26 = 28th move and do another at 2 + 26*2 = 54th move, 
and so on.
Need to find maximum move we need and make sure that one is <= k.
Since we can pick any index to shift, so the order doesn’t matter.
We can start from left to right.

Time complexity: O(n)
Space complexity: O(26) = O(1)*/
public class Solution {
    public bool CanConvertString(string s, string t, int k) {
        if (s.Length != t.Length) return false;
    int[] count = new int[26];    
    for (int i = 0; i < s.Length; ++i) {            
      int d = (t[i] - s[i] + 26) % 26;
      int c = count[d]++;
      if (d != 0 && d + c * 26 > k)
        return false;
    }    
    return true;
    }
}

// 1503. Last Moment Before All Ants Fall Out of a Plank
/*Solution: Keep Walking
When two ants A –> and <– B meet at some point, 
they change directions <– A B –>, 
we can swap the ids of the ants as <– B A–>, 
so it’s the same as walking individually and passed by. 
Then we just need to find the max/min of the left/right arrays.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int GetLastMoment(int n, int[] left, int[] right) {
        int t1 = left.Length == 0 ? 0 : left.Max();
    int t2 = right.Length == 0 ? 0 : n - right.Min();
    return Math.Max(t1, t2);
    }
}

// 1504. Count Submatrices With All Ones
/*Solution 1: Brute Force w/ Pruning
Time complexity: O(m^2*n^2)
Space complexity: O(1)

*/
public class Solution {
    public int NumSubmat(int[][] mat) {
         int R = mat.Length;
    int C = mat[0].Length;
    // # of sub matrices with top-left at (sr, sc)
    int subMats(int sr, int sc) {
      int max_c = C;
      int count = 0;
      for (int r = sr; r < R; ++r)
        for (int c = sc; c < max_c; ++c)
          if (mat[r][c] > 0) {
            ++count;
          } else {
            max_c = c;
          }
      return count;
    };    
    int ans = 0;
    for (int r = 0; r < R; ++r)
      for (int c = 0; c < C; ++c)
        ans += subMats(r, c);
    return ans;
    }
}

// 1505. Minimum Possible Integer After at Most K Adjacent Swaps On Digits
/*Solution 2: Binary Indexed Tree / Fenwick Tree

Moving elements in a string is a very expensive operation, 
basically O(n) per op. Actually, we don’t need to move the elements physically, 
instead we track how many elements before i has been moved to the “front”. 
Thus we know the cost to move the i-th element to the “front”, 
which is i – elements_moved_before_i or prefix_sum(0~i-1) 
if we mark moved element as 1.

We know BIT / Fenwick Tree is good for dynamic prefix sum computation 
which helps to reduce the time complexity to O(nlogn).

Time complexity: O(nlogn)
Space complexity: O(n)*/
public class Solution {
    public string MinInteger(string num, int k) {
         int n = num.Length;
    var used = new bool[n];
    var pos = new List<Queue<int>>(10);
    for (int i = 0; i <= 9; ++i)
      pos.Add(new Queue<int>());    
    for (int i = 0; i < num.Length; ++i)
      pos[num[i] - '0'].Enqueue(i);
    var tree = new Fenwick(n);
    var sb = new StringBuilder();
    while (k > 0 && sb.Length < n) {
      for (int d = 0; d <= 9; ++d) {
        if (pos[d].Count == 0) continue;
        int i = pos[d].Peek();
        int cost = i - tree.query(i - 1);
        if (cost > k) continue;        
        k -= cost;
        sb.Append((char)(d + '0'));
        tree.update(i, 1);
        used[i] = true;
        pos[d].Dequeue();
        break;
      }
    }
    for (int i = 0; i < n; ++i)
      if (!used[i]) sb.Append(num[i]);
    return sb.ToString();
    
    }
    
 
  public class Fenwick {
    private int[] sums;
    
    public Fenwick(int n) {
      this.sums = new int[n + 1];
    }
    
    public void update(int i, int delta) {
      ++i;
      while (i < sums.Length) {
        this.sums[i] += delta;
        i += i & -i;
      }
    }
    
    public int query(int i) {
      int ans = 0;
      ++i;
      while (i > 0) {
        ans += this.sums[i];
        i -= i & -i;
      }
      return ans;
    }
  }
  
}

// 1494. Parallel Courses II
/*Solution: DP / Bitmask
NOTE: This is a NP problem, any polynomial-time algorithm is incorrect 
otherwise P = NP.

Variant 1:
dp[m] := whether state m is reachable, where m is the bitmask of courses studied.
For each semester, we enumerate all possible states from 0 to 2^n – 1, 
if that state is reachable, then we choose c (c <= k) courses from n and 
check whether we can study those courses.
If we can study those courses, we have a new reachable state, 
we set dp[m | courses] = true.

Time complexity: O(n*2^n*2^n) = O(n*n^4) <– This will be much smaller in practice.
and can be reduced to O(n*3^n).
Space complexity: O(2^n)*/
public class Solution {
    public int MinNumberOfSemesters(int n, int[][] relations, int k) {
        
          int S = 1 << n;
    int[] deps  = new int[n];//(n); // deps[i] = dependency mask for course i.
    foreach (int[] d in relations)
      deps[d[1] - 1] |= (1 << (d[0] - 1));    
    // dp[m] := min semesters to reach state m.
    
    int[] dp = new int[S];
    dp[0] = 1;
   
    
    for (int d = 1; d <= n; ++d) { // at most n semesters.
      int[] tmp = new int[S];// start a new semesters.
      for (int s = 0; s < S; ++s) {
        if (dp[s] == 0) continue; // not a reachable state.
        int mask = 0;
        for (int i = 0; i < n; ++i)
          if ((s & (1 << i)) == 0 && (s & deps[i]) == deps[i]) 
            mask |= (1 << i);
        // Prunning, take all.
        if (BitOperations.PopCount((uint)mask) <= k) {
          tmp[s | mask] = 1;         
        } else {
          // Try all subsets. 
          for (int c = mask; c > 0; c = (c - 1) & mask)
            if (BitOperations.PopCount((uint)c) <= k) {
              tmp[s | c] = 1;
            }
        }
        if (tmp.Last() != 0) return d;
      }
     // dp.swap(tmp);     
        int[] t = dp;
        dp = tmp;
        tmp = t;
    }
    return -1;
    }
    
}

// 1496. Path Crossing
/*Solution: Simulation + Hashtable
Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public bool IsPathCrossing(string path) {
        HashSet<int> s = new HashSet<int>();
    int x = 0;
    int y = 0;
    s.Add(x * 10000 + y);
    foreach (char d in path) {
      if (d == 'N') ++y;
      else if (d == 'S') --y;
      else if (d == 'E') ++x;
      else if (d == 'W') --x;
      if (!s.Add(x * 10000 + y))
        return true;
    }
    return false;
    }
}

// 1497. Check If Array Pairs Are Divisible by k
/*Solution: Mod and Count
Count the frequency of (x % k + k) % k.
f[0] should be even (zero is also even)
f[1] = f[k -1] ((1 + k – 1) % k == 0)
f[2] = f[k -2] ((2 + k – 2) % k == 0)
…
Time complexity: O(n)
Space complexity: O(k)

*/
public class Solution {
    public bool CanArrange(int[] arr, int k) {
        int[] f = new int[k];//(k);
    foreach (int x in arr) ++f[(x % k + k) % k];
    for (int i = 1; i < k; ++i)
      if (f[i] != f[k - i]) return false;
    return f[0] % 2 == 0;
    }
}

// 1499. Max Value of Equation
/*Solution 2: Monotonic Queue

Maintain a monotonic queue:
1. The queue is sorted by y – x in descending order.
2. Pop then front element when xj – x_front > k, they can’t be used anymore.
3. Record the max of {xj + yj + (y_front – x_front)}
4. Pop the back element when yj – xj > y_back – x_back, 
they are smaller and lefter. Won’t be useful anymore.
5. Finally, push the j-th element onto the queue.

Time complexity: O(n)
Space complexity: O(n)

*/
public class Solution {
    public int FindMaxValueOfEquation(int[][] points, int k) {
        var q = new LinkedList<int>();
    int ans = Int32.MinValue;
    for (int i = 0; i < points.Length; ++i) {
      int xj = points[i][0];
      int yj = points[i][1];
      
      while (q.Count != 0 && xj - points[q.First.Value][0] > k) {
        q.RemoveFirst();
      }
      
      if (q.Count != 0) {
        ans = Math.Max(ans, xj + yj 
                + points[q.First.Value][1] - points[q.First.Value][0]);
      }
      // remember to q.Last.Value is so much qicker than q.Last()
        // same for q.First.Value instead of q.First().
      while (q.Count != 0 && yj - xj 
                >= points[q.Last.Value][1] - points[q.Last.Value][0]) {
        q.RemoveLast();
      }
      
      q.AddLast(i);
    }
    return ans;
    }
}
/*Observation
Since xj > xi, so |xi – xj| + yi + yj => xj + yj + (yi – xi)
We want to have yi – xi as large as possible while need to make sure xj – xi <= k.

Solution 1: Priority Queue / Heap
Put all the points processed so far onto the heap as (y-x, x) 
sorted by y-x in descending order.
Each new point (x_j, y_j), find the largest y-x such that x_j – x <= k.

Time complexity: O(nlogn)
Space complexity: O(n)

*/
public class Solution {
    public int FindMaxValueOfEquation(int[][] points, int k) {
       PriorityQueue<KeyValuePair<int, int>,KeyValuePair<int, int>> pq = new PriorityQueue<KeyValuePair<int, int>,KeyValuePair<int, int>>(Comparer<KeyValuePair<int,int>>.Create((a, b) => a.Key == b.Key ? a.Value - b.Value : b.Key - a.Key));
        int res = Int32.MinValue;
        foreach (int[] point in points) {
            while (pq.Count != 0 && point[0] - pq.Peek().Value > k) {
                pq.Dequeue();
            }
            if (pq.Count != 0) {
                res = Math.Max(res, pq.Peek().Key + point[0] + point[1]);
            }
            pq.Enqueue(new KeyValuePair<int, int>(point[1] - point[0], point[0]),new KeyValuePair<int, int>(point[1] - point[0], point[0]));
        }
        return res;
    }
}

// 1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree
/*Solution: Brute Force?

For each edge
1. exclude it and build a MST, cost increased => critical
2. for a non critical edge, force include it and build a MST, 
cost remains the same => pseudo critical

Proof of 2, if a non critical / non pseudo critical edge was added into the MST, 
the total cost must be increased. 
So if the cost remains the same, must be the other case. 
Since we know the edge is non-critical, so it has to be pseudo critical.*/
public class UnionFind {
 public UnionFind(int n){ 
     //iota(begin(p_), end(p_), 0); 
    // : p_(n), r_(n) 
     r_ = new int[ n ]; //int[] p_ = new int[ n ];
 p_ = Enumerable.Range(0, n).ToArray();
 }// e.g. p[i] = i
 public int Find(int x) { return p_[x] == x ? x : p_[x] = Find(p_[x]); }
  public bool Union(int x, int y) {
    int rx = Find(x);
    int ry = Find(y);
    if (rx == ry) return false;
    if (r_[rx] == r_[ry]) {
      p_[rx] = ry;
      ++r_[ry];
    } else if (r_[rx] > r_[ry]) {
      p_[ry] = rx;
    } else {
      p_[rx] = ry;
    }    
    return true;
  }
 private int[] p_;  
    private int[] r_; 
};
 

public class Solution {
    public IList<IList<int>> FindCriticalAndPseudoCriticalEdges(int n, int[][] edges) {
        // Record the original id.
    for (int i = 0; i < edges.Length; ++i) {
       
        var t = edges[i].ToList();t.Add(i);
        edges[i] = t.ToArray();

    }
        //edges[i].push_back(i);
    // Sort edges by weight.
    /*sort(begin(edges), end(edges), [&](const auto& e1, const auto& e2){
      if (e1[2] != e2[2]) return e1[2] < e2[2];        
      return e1 < e2;
    });*/
    Array.Sort(edges, (a,b) => a[2] - b[2] );
    // Cost of MST, ex: edge to exclude, in: edge to include.
    int MST( int ex = -1, int index = -1) {
       
      UnionFind uf = new UnionFind(n);
      int cost = 0;
      int count = 0;
      if (index >= 0) {
        cost += edges[index][2];
        uf.Union(edges[index][0], edges[index][1]);
        count++;
      }
      for (int i = 0; i < edges.Length; ++i) {        
        if (i == ex) continue;
        if (!uf.Union(edges[i][0], edges[i][1])) continue;
        cost += edges[i][2];
        ++count;
      }
      return count == n - 1 ? cost : Int32.MaxValue;
    };
     int min_cost = MST();
    List<int> criticals = new List<int>();
    List<int> pseudos = new List<int>();
    for (int i = 0; i < edges.Length; ++i) {
      // Cost increased or can't form a tree.
      if (MST(i) > min_cost) {
        criticals.Add(edges[i][3]);
      } else if (MST(-1, i) == min_cost) {
        pseudos.Add(edges[i][3]);
      }
    }
    return new List<IList<int>>(){criticals, pseudos};
    }
}

// 1492. The kth Factor of n
/*Solution: Brute Force
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int KthFactor(int n, int k) {
        for (int i = 1; i <= n; ++i)
      if (n % i == 0 && --k == 0) return i;
    return -1;
    }
}

// 1493. Longest Subarray of 1's After Deleting One Element
/*Solution 3: Sliding Window

Maintain a sliding window l ~ r s.t sum(num[l~r]) >= r – l. 
There can be at most one 0 in the window.
ans = max{r – l} for all valid windows.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int LongestSubarray(int[] nums) {
         int n = nums.Length;
    int ans = 0;
    int sum = 0; // sum of nums[l~r].
    for (int l = 0, r = 0; r < n; ++r) {
      sum += nums[r];
      // Maintain sum >= r - l, at most 1 zero.
      while (l < r && sum < r - l)
        sum -= nums[l++];
      ans = Math.Max(ans, r - l);
    }
    return ans;
    }
}

// 1487. Making File Names Unique
/*Solution: Hashtable
Use a hashtable to store the mapping form base_name to its next suffix index.

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public string[] GetFolderNames(string[] names) {
        List<string> ans = new List<string>();
    Dictionary<string, int> m = new Dictionary<string, int>(); // base_name -> next suffix.
    foreach (string name in names) {
        string unique_name = name;
        if(m.ContainsKey(name)) {
      
      int j = m[name];
      if (j > 0) {
        while (m.ContainsKey(unique_name = name + "(" + (j++).ToString() + ")"));    
        m[name] = j;        
      }
        
        }
       m.Add(unique_name, 1); //m[unique_name] = 1;
      ans.Add(unique_name);
    }
    return ans.ToArray();
    }
}

// 1486. XOR Operation in an Array
/*Solution: Simulation
Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int XorOperation(int n, int start) {
        int ans = 0;
    for (int i = 0; i < n; ++i)
      ans ^= (start + 2 * i);
    return ans;
    }
}

// 1481. Least Number of Unique Integers after K Removals
/*Solution: Greedy
Count the frequency of each unique number. 
Sort by frequency, remove items with lowest frequency first.

Time complexity: O(nlogn)
Space complexity: O(n)

*/
public class Solution {
    public int FindLeastNumOfUniqueInts(int[] arr, int k) {
         Dictionary<int, int> c = new Dictionary<int, int>();
    foreach (int x in arr) c[x] = c.GetValueOrDefault(x, 0) +1;
    List<int> m = new List<int>(); // freq
    foreach (var (x, f) in c)
      m.Add(f);
    m.Sort();//sort(begin(m), end(m));
    int ans = m.Count;    
    int i = 0;
    while ((k--) > 0) {
      if (--m[i] == 0) {
        ++i;
        --ans;
      }
    }
    return ans;
    }
}

// 1478. Allocate Mailboxes
/*DP Solution

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

Note that solution O(KN) is also possible to come up with.*/
public class Solution {
    public int MinDistance(int[] A, int K) {
         Array.Sort(A);
        int n = A.Length; int[] B = new int[n+1]; int[] dp = new int[n];
        for (int i = 0; i < n; ++i) {
            B[i + 1] = B[i] + A[i];
            dp[i] = (int)1e6;
        }
        for (int k = 1; k <= K; ++k) {
            for (int j = n - 1; j > k - 2; --j) {
                for (int i = k - 2; i < j; ++i) {
                    int m1 =  (i + j + 1) / 2, m2 = (i + j + 2) / 2;
                    int last = (B[j + 1] - B[m2]) - (B[m1 + 1] - B[i + 1]);
                    dp[j] = Math.Min(dp[j], (i >= 0 ? dp[i] : 0) + last);
                }
            }
        }
        return dp[n - 1];
    }
}

// 1476. Subrectangle Queries
/*Solution 2: Geometry
For each update remember the region and value.

For each query, find the newest updates that covers the query point. 
If not found, return the original value in the matrix.

Time complexity:
Update: O(1)
Query: O(|U|), where |U| is the number of updates so far.

Space complexity: O(|U|)

*/
public class SubrectangleQueries {

    public SubrectangleQueries(int[][] rectangle) {
         m_ = rectangle;
    }
    
    public void UpdateSubrectangle(int row1, int col1, int row2, int col2, int newValue) {
        updates_.AddFirst(new int[5]{row1, col1, row2, col2, newValue});
    }
    
    public int GetValue(int row, int col) {
        foreach (int[] u in updates_)
          if (row >= u[0] && row <= u[2] && col >= u[1] && col <= u[3])
            return u[4];
    return m_[row][col];
    }
    
  
private int[][] m_ ;
 private LinkedList<int[]> updates_ = new LinkedList<int[]>();  
}

/**
 * Your SubrectangleQueries object will be instantiated and called as such:
 * SubrectangleQueries obj = new SubrectangleQueries(rectangle);
 * obj.UpdateSubrectangle(row1,col1,row2,col2,newValue);
 * int param_2 = obj.GetValue(row,col);
 */

// 1477. Find Two Non-overlapping Sub-arrays Each With Target Sum
/*Solution: Sliding Window + Best so far


Use a sliding window to maintain a subarray whose sum is <= target
When the sum of the sliding window equals to target, we found a subarray [s, e]
Update ans with it’s length + shortest subarray which ends before s.
We can use an array to store the shortest subarray which ends before s.
Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int MinSumOfLengths(int[] arr, int target) {
        int kInf = (int)1e9;
    int n = arr.Length;
    // min_lens[i] := min length of a valid subarray ends or before i.
    int[] min_lens = new int[n];Array.Fill(min_lens, kInf);
    int ans = kInf;
    int sum = 0;
    int s = 0;
    int min_len = kInf;
    for (int e = 0; e < n; ++e) {
      sum += arr[e];
      while (sum > target) sum -= arr[s++];
      if (sum == target) {       
        int cur_len = e - s + 1;
        if (s > 0 && min_lens[s - 1] != kInf)
          ans = Math.Min(ans, cur_len + min_lens[s - 1]);
        min_len = Math.Min(min_len, cur_len);
      }
      min_lens[e] = min_len;
    }    
    return ans >= kInf ? -1 : ans;
    }
}

// 1475. Final Prices With a Special Discount in a Shop
/*Solution 2: Monotonic Stack

Use a stack to store monotonically increasing items, 
when the current item is cheaper than the top of the stack, 
we get the discount and pop that item. 
Repeat until the current item is no longer cheaper or the stack becomes empty.

Time complexity: O(n)
Space complexity: O(n)

index version
*/
public class Solution {
    public int[] FinalPrices(int[] prices) {
        // stores indices of monotonically incraseing elements.
    Stack<int> s = new Stack<int>(); 
    for (int i = 0; i < prices.Length; ++i) {
      while (s.Count != 0 && prices[s.Peek()] >= prices[i]) {
        prices[s.Peek()] -= prices[i];
        s.Pop();
      }
      s.Push(i);
    }      
    return prices;
    }
}

// 1537. Get the Maximum Score
/*Solution: Two Pointers + DP
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
Space complexity: O(n) -> O(1)'''*/
public class Solution {
    public int MaxSum(int[] nums1, int[] nums2) {
        int kMod = 1_000_000_007;
    int n1 = nums1.Length;
    int n2 = nums2.Length;
    int i = 0;
    int j = 0;
    long a = 0;
    long b = 0;
    while (i < n1 || j < n2) {
      if (i < n1 && j < n2 && nums1[i] == nums2[j]) {
        a = b = Math.Max(a, b) + nums1[i];
        ++i;
        ++j;
      } else if (i < n1 && (j == n2 || nums1[i] < nums2[j])) {
        a += nums1[i++];        
      } else {
        b += nums2[j++];
      }
    }
    return (int)(Math.Max(a, b) % kMod);
    }
}

// 1536. Minimum Swaps to Arrange a Binary Grid
/*Solution: Bubble Sort
Counting how many tailing zeros each row has.
Then input
[0, 0, 1]
[1, 1, 0]
[1, 0, 0]
becomes [0, 1, 2]
For i-th row, it needs n – i – 1 tailing zeros.
We need to find the first row that has at least n – i – 1 
tailing zeros and bubbling it up to the i-th row. 
This process is very similar to bubble sort.
[0, 1, 2] -> [0, 2, 1] -> [2, 0, 1]
[2, 0, 1] -> [2, 1, 0]
Total 3 swaps.

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int MinSwaps(int[][] grid) {
        int n = grid.Length;
    List<int> zeros = new List<int>();
    foreach (var row in grid) {
      int zero = 0;
      for (int i = n - 1; i >= 0 && row[i] == 0; --i) 
        ++zero;
      zeros.Add(zero);
    }
    int ans = 0;
    for (int i = 0; i < n; ++i) {
      int j = i;
      int z = n - i - 1; // the i-th needs n - i - 1 zeros
      // Find first row with at least z tailing zeros.
      while (j < n && zeros[j] < z) ++j;
      if (j == n) return -1;
      while (j > i) {
        zeros[j] = zeros[j - 1];
        --j;
        ++ans;
      }
    }
    return ans;
    }
}

// 1535. Find the Winner of an Array Game
/*Solution 2: One pass
Since smaller numbers will be put to the end of the array, 
we must compare the rest of array before meeting those used numbers again. 
And the winner is monotonically increasing. So we can do it in one pass, 
just keep the largest number seen so far. If we reach to the end of the array, 
arr[0] will be max(arr) and it will always win no matter what k is.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int GetWinner(int[] arr, int k) {
        int winner = arr[0];
        int win = 0;
        for (int i = 1; i < arr.Length && win < k; ++i, ++win)
          if (arr[i] > winner) {
            winner = arr[i];
            win = 0;
          }
        return winner;
    }
}

// 1534. Count Good Triplets
/*Solution: Brute Force
Time complexity: O(n^3)
Space complexity: O(1)*/
public class Solution {
    public int CountGoodTriplets(int[] arr, int a, int b, int c) {
        int n = arr.Length;
    int ans = 0;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        for (int k = j + 1; k < n; ++k)
          if (Math.Abs(arr[i] - arr[j]) <= a &&
              Math.Abs(arr[j] - arr[k]) <= b &&
              Math.Abs(arr[i] - arr[k]) <= c)
            ++ans;
    return ans;
    }
}

// 1528. Shuffle String
/*Solution: Simulation
Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public string RestoreString(string s, int[] indices) {
        char[] ans = new char[s.Length];
        for (int i = 0; i < s.Length; ++i)
          ans[indices[i]] = s[i];
        return new String(ans);
    }
}

// 1530. Number of Good Leaf Nodes Pairs
/*Solution 2: Post order traversal

For each node, compute the # of good leaf pair under itself.
1. count the frequency of leaf node at distance 1, 2, …, d 
for both left and right child.
2. ans += l[i] * r[j] (i + j <= distance) cartesian product
3. increase the distance by 1 for each leaf node when pop
Time complexity: O(n*D^2)
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
    public int CountPairs(TreeNode root, int distance) {
        this.D = distance;
    this.ans = 0;
    dfs(root);
    return ans;
    }
    
    private int D;
  private int ans;
  
  private int[] dfs(TreeNode root) {
    int[] f = new int[D + 1];
    if (root == null) return f;
    if (root.left == null && root.right == null) {
      f[0] = 1;
      return f;
    }
    int[] fl = dfs(root.left);
    int[] fr = dfs(root.right);
    for (int i = 0; i + 1 <= D; ++i)
      for (int j = 0; i + j + 2 <= D; ++j)
        this.ans += fl[i] * fr[j];
    for (int i = 0; i < D; ++i)
      f[i + 1] = fl[i] + fr[i];
    return f;
  }
}

// 1529. Minimum Suffix Flips
/*Solution: XOR
Flip from left to right, since flipping the a bulb won’t affect anything in the left.
We count how many times flipped so far, and that % 2 will be the state for all the bulb to the right.
If the current bulb’s state != target, we have to flip once.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MinFlips(string target) {
        int ans = 0;
    int cur = 0;
    foreach (char c in target) {
      if (c - '0' != cur) {
        cur ^= 1;
        ++ans;
      }
    }
    return ans;
    }
}

// 1531. String Compression II
/*State compression

dp[i][k] := min len of s[i:] encoded by deleting at most k charchters.

dp[i][k] = min(dp[i+1][k-1] # delete s[i]
encode_len(s[i~j] == s[i]) + dp(j+1, k – sum(s[i~j])) for j in range(i, n)) # keep

Time complexity: O(n^2*k)
Space complexity: O(n*k)*/
public class Solution {
    public int GetLengthOfOptimalCompression(string s, int k) {
 
    this.s = s.ToCharArray();
    this.n = s.Length;
    this.dp_ = new int[n][];//[k + 1];
    
    for (int i = 0; i < n; i++){
        this.dp_[i] = new int[k + 1];
        Array.Fill(this.dp_[i], -1);
    }
      
    return dp(0, k);
    }
    
    private int[][] dp_;
  private char[] s;
  private int n;
  
  private int dp(int i, int k) {
    if (k < 0) return this.n;
    if (i + k >= n) return 0; // done or delete all.    
    int ans = dp_[i][k];
    if (ans != -1) return ans;
    ans = dp(i + 1, k - 1); // delete s[i]
    int len = 0;
    int same = 0;    
    int diff = 0;
    for (int j = i; j < n && diff <= k; ++j) {
      if (s[j] == s[i]) {
        ++same;
        if (same <= 2 || same == 10 || same == 100) ++len;
      } else {
        ++diff;
      }      
      ans = Math.Min(ans, len + dp(j + 1, k - diff)); 
    }
    dp_[i][k] = ans;
    return ans;
}
    
}

// 1513. Number of Substrings With Only 1s
/*Solution: DP / Prefix Sum
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

dp[i] only depends on dp[i-1], we can reduce the space complexity to O(1)*/
public class Solution {
    public int NumSub(string s) {
    int kMod = 1_000_000_000 + 7;
    int ans = 0;
    int cur = 0;
    for (int i = 0; i < s.Length; ++i) {
      cur = s[i] == '1' ? cur + 1 : 0;
      ans = (ans + cur) % kMod;
    }
    return ans;
    }
}

// 1521. Find a Value of a Mysterious Function Closest to Target
/*There is another HashSet O(32 * N) ==> O(N) solution, though space is not strictly O(1) :*/
public class Solution {
    public int ClosestToTarget(int[] arr, int target) {
         int m = arr.Length; int res = Int32.MaxValue;//Integer.MAX_VALUE;
        HashSet<int> s = new HashSet<int>();
        for (int i = 0; i < m; i++) {
           HashSet<int> tmp = new HashSet<int>();
            tmp.Add(arr[i]);
            foreach (int n in s)  tmp.Add(n & arr[i]);
            foreach (int n in tmp) res = Math.Min(res, Math.Abs(n - target));
            s = tmp;
        }
        return res;
    }
}

// 1524. Number of Sub-arrays With Odd Sum
/*Solution: DP

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

*/
public class Solution {
    public int NumOfSubarrays(int[] arr) {
        long ans = 0, odd = 0, even = 0;
    foreach (int x in arr) {
      even += 1;
      if (x % 2 == 1) {
        long t = even; even = odd; odd = t;
      }
      ans += odd;
    }
    return (int)(ans % (int)(1e9 + 7));
    }
}

// 1525. Number of Good Ways to Split a String
/*Solution: Sliding Window
Count the frequency of each letter and count number of unique letters 
for the entire string as right part.
Iterate over the string, add current letter to the left part, 
and remove it from the right part.
We only
increase the number of unique letters when its frequency becomes to 1
decrease the number of unique letters when its frequency becomes to 0
Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int NumSplits(string s) {
        int[] l = new int[26];
    int[] r = new int[26];
    int ans = 0;
    int cl = 0;
    int cr = 0;
    foreach (char c in s.ToCharArray())
      if (++r[c - 'a'] == 1) ++cr;
    foreach (char c in s.ToCharArray()) {
      if (++l[c - 'a'] == 1) ++cl;
      if (--r[c - 'a'] == 0) --cr;
      if (cl == cr) ++ans;
    }
    return ans;
    }
}

// 1526. Minimum Number of Increments on Subarrays to Form a Target Array
/*Solution:
If arr[i] < arr[i – 1], if we can generate arr[i] 
then we can always generate arr[i] with no extra cost.
e.g. [3, 2, 1] 3 < 2 < 1, [0,0,0] => [1,1,1] => [2, 2, 1] => [3, 2, 1]

So we only need to handle the case of arr[i] > arr[i – 1], 
we can reuse the computation, with extra cost of arr[i] – arr[i-1].
e.g. [2, 5]: [0,0] => [1, 1] => [2, 2], it takes 2 steps to cover arr[0].
[2,2] => [2, 3] => [2, 4] => [2, 5], 
takes another 3 steps (arr[1] – arr[0] / 5 – 2) to cover arr[1].

Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public int MinNumberOperations(int[] target) {
        int ans = target[0];
    for (int i = 1; i < target.Length; ++i)
      ans += Math.Max(0, target[i] - target[i - 1]);
    return ans;
    }
}

// 1510. Stone Game IV
/*Solution: Recursion w/ Memoization / DP
Let win(n) denotes whether the current play will win or not.
Try all possible square numbers and see whether the other player will lose or not.
win(n) = any(win(n – i*i) == False) ? True : False
base case: win(0) = False

Time complexity: O(nsqrt(n))
Space complexity: O(n)*/
public class Solution {
    public bool WinnerSquareGame(int n) {
        this.cache = new int[n + 1];
    return this.win(n) > 0;
    }
    
    private int[] cache;
  
  private int win(int n) {
    if (n == 0) return -1;
    if (this.cache[n] != 0) return this.cache[n];
    for (int i = (int)Math.Sqrt(n); i >= 1; --i)
      if (win(n - i * i) < 0) 
        return this.cache[n] = 1;
    return this.cache[n] = -1;
  }

}

// 1518. Water Bottles
/*Solution: Simulation
Time complexity: O(logb/loge)?
Space complexity: O(1)*/
public class Solution {
    public int NumWaterBottles(int numBottles, int numExchange) {
        int ans = numBottles;
        while (numBottles >= numExchange) {
            int remainder = numBottles % numExchange;
            numBottles /= numExchange;
            ans += numBottles;
            numBottles += remainder;
        }
        return ans;
    }
}

public class Solution {
    public int NumWaterBottles(int numBottles, int numExchange) {
       return numBottles + (numBottles - 1) / (numExchange - 1);
    }
}

// 1519. Number of Nodes in the Sub-Tree With the Same Label
/*Solution: Post order traversal + hashtable
For each label, record the count. 
When visiting a node, we first record the current count of its label as before, 
and traverse its children, when done, increment the current count, 
ans[i] = current – before.

Time complexity: O(n)
Space complexity: O(n)*/
public class Solution {
    public int[] CountSubTrees(int n, int[][] edges, string labels) {
        this.g = new List<List<int>>(n);
    this.labels = labels; 
    for (int i = 0; i < n; ++i)
      this.g.Add(new List<int>());
    foreach (int[] e in edges) {
      this.g[e[0]].Add(e[1]);
      this.g[e[1]].Add(e[0]);
    }
    this.ans = new int[n];
    this.seen = new int[n];
    this.count = new int[26];
    this.postOrder(0);
    return ans;
    }
    
    private List<List<int>> g;
  private String labels;
  private int[] ans;
  private int[] seen;
  private int[] count;
  
  
  private void postOrder(int i) {
    if (this.seen[i]++ > 0) return;
    int before = this.count[this.labels[i] - 'a'];
    foreach (int j in this.g[i]) this.postOrder(j);
    this.ans[i] = ++this.count[this.labels[i] - 'a'] - before;    
  }
}

// 1520. Maximum Number of Non-Overlapping Substrings
/*Solution: Greedy
Observation: If a valid substring contains shorter valid strings, ignore the longer one and use the shorter one.
e.g. “abbeefba” is a valid substring, however, it includes “bbeefb”, “ee”, “f” three valid substrings, thus it won’t be part of the optimal solution, since we can always choose a shorter one, with potential to have one or more non-overlapping substrings. For “bbeefb”, again it includes “ee” and “f”, so it won’t be optimal either. Thus, the optimal ones are “ee” and “f”.

We just need to record the first and last occurrence of each character
When we meet a character for the first time we must include everything from current pos to it’s last position. e.g. “abbeefba” | ccc, from first ‘a’ to last ‘a’, we need to cover “abbeefba”
If any character in that range has larger end position, we must extend the string. e.g. “abcabbcc” | efg, from first ‘a’ to last ‘a’, we have characters ‘b’ and ‘c’, so we have to extend the string to cover all ‘b’s and ‘c’s. Our first valid substring extended from “abca” to “abcabbcc”.
If any character in the covered range has a smallest first occurrence, then it’s an invalid substring. e.g. ab | “cbc”, from first ‘c’ to last ‘c’, we have ‘b’, but ‘b’ is not fully covered, thus “cbc” is an invalid substring.
For the first valid substring, we append it to the ans array. “abbeefba” => ans = [“abbeefba”]
If we find a shorter substring that is full covered by the previous valid substring, we replace that substring with the shorter one. e.g.
“abbeefba” | ccc => ans = [“abbeefba”]
“abbeefba” | ccc => ans = [“bbeefb”]
“abbeefba” | ccc => ans = [“ee”]
If the current substring does not overlap with previous one, append it to ans array.
“abbeefba” | ccc => ans = [“ee”]
“abbeefba” | ccc => ans = [“ee”, “f”]
“abbeefbaccc” => ans = [“ee”, “f”, “ccc”]
Time complexity: O(n)
Space complexity: O(1)

*/
public class Solution {
    public IList<string> MaxNumOfSubstrings(string s) {
        int n = s.Length;    
    int[] l = new int[26];Array.Fill(l, Int32.MaxValue);//(26, INT_MAX);
    int[] r = new int[26];Array.Fill(r, Int32.MinValue);
    for (int i = 0; i < n; ++i) {
      l[s[i] - 'a'] = Math.Min(l[s[i] - 'a'], i);
      r[s[i] - 'a'] = Math.Max(r[s[i] - 'a'], i);
    }
    int extend(int i) {      
      int p = r[s[i] - 'a'];
      for (int j = i; j <= p; ++j) {
        if (l[s[j] - 'a'] < i) // invalid substring
          return -1; // e.g. a|"ba"...b
        p = Math.Max(p, r[s[j] - 'a']);
      }
      return p;
    };
    
     IList<string> ans = new List<string>();
    int last = -1;
    for (int i = 0; i < n; ++i) {
      if (i != l[s[i] - 'a']) continue;
      int p = extend(i);
      if (p == -1) continue;
      if (i > last) ans.Add("");
      ans[ans.Count - 1] = s.Substring(i, p - i + 1);
      last = p;      
    }
    return ans;
    }
}

// 1514. Path with Maximum Probability
/*Solution: Dijkstra’s Algorithm
max(P1*P2*…*Pn) => max(log(P1*P2…*Pn)) 
=> max(log(P1) + log(P2) + … + log(Pn) => min(-(log(P1) + log(P2) … + log(Pn)).

Thus we can convert this problem to the classic single source shortest path problem 
that can be solved with Dijkstra’s algorithm.

Time complexity: O(ElogV)
Space complexity: O(E+V)*/
public class Solution {
    public double MaxProbability(int n, int[][] edges, double[] succProb, int start, int end) {
          List<List<KeyValuePair<int, double>>> g = new List<List<KeyValuePair<int, double>>>(n); // u -> {v, -log(w)}
        for (int i = 0; i < n; ++i) g.Add(new List<KeyValuePair<int, double>>());
    for (int i = 0; i < edges.Length; ++i) {
      double w = -Math.Log(succProb[i]);
      g[edges[i][0]].Add(new KeyValuePair<int, double>(edges[i][1], w));
      g[edges[i][1]].Add(new KeyValuePair<int, double>(edges[i][0], w));
    }
 
    //vector<double> dist(n, FLT_MAX);
   // priority_queue<pair<double, int>> q;
   // q.emplace(-0.0, start);
   // vector<int> seen(n);
        
    var seen = new int[n];
    var dist = new double[n];
    Array.Fill(dist, Double.MaxValue);

    // {u, dist[u]}
    var q = new PriorityQueue<KeyValuePair<double, int>, KeyValuePair<double, int>>(Comparer<KeyValuePair<double, int>>.Create((a, b) => b.Key.CompareTo(a.Key)));    
    q.Enqueue(new KeyValuePair<double,int>(-0.0, start), new KeyValuePair<double,int>(-0.0, start));
        
    while (q.Count != 0) {
      double d = -q.Peek().Key;
      int u = q.Peek().Value;        
      q.Dequeue();
      seen[u] = 1;
      if (u == end) return Math.Exp(-d);
      foreach (var (v, w) in g[u]) {
        if (seen[v] > 0 || d + w > dist[v]) continue;
        q.Enqueue(new KeyValuePair<double,int>(-(dist[v] = d + w), v), new KeyValuePair<double,int>(-(dist[v] = d + w), v));
      }
    }
    return 0;
    }
}

// 1512. Number of Good Pairs
/*Solution 2: Hashtable
Store the frequency of each number so far, when we have a number x at pos j, 
and it appears k times before. Then we can form additional k pairs.

Time complexity: O(n)
Space complexity: O(range)*/
public class Solution {
    public int NumIdenticalPairs(int[] nums) {
         int ans = 0;
    int[] f = new int[101];
    for (int i = 0; i < nums.Length; ++i)      
      ans += f[nums[i]]++;
    return ans;
    }
}

// 1507. Reformat Date
/*Solution: String + HashTable
Time complexity: O(1)
Space complexity: O(1)

*/
public class Solution {
    public string ReformatDate(string date) {
        Dictionary<String, String> m = new  Dictionary<String, String>();
    m.Add("Jan", "01");
    m.Add("Feb", "02");
    m.Add("Mar", "03");
    m.Add("Apr", "04");
    m.Add("May", "05");
    m.Add("Jun", "06");
    m.Add("Jul", "07");
    m.Add("Aug", "08");
    m.Add("Sep", "09");
    m.Add("Oct", "10");
    m.Add("Nov", "11");
    m.Add("Dec", "12");
    String[] items = date.Split(" ");    
    String day = items[0].Substring(0, items[0].Length - 2);
    if (day.Length == 1) day = "0" + day;
    return items[2] + "-" + m[items[1]] + "-" + day;
    }
}

// 662. Maximum Width of Binary Tree
/*Solution: DFS

Let us assign an id to each node, similar to the index of a heap. 
root is 1, left child = parent * 2, right child = parent * 2 + 1. 
Width = id(right most child) – id(left most child) + 1, so far so good.
However, this kind of id system grows exponentially, 
it overflows even with long type with just 64 levels. 
To avoid that, we can remap the id with id – id(left most child of each level).

Time complexity: O(n)
Space complexity: O(h)*/
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
    public int WidthOfBinaryTree(TreeNode root) {
         this.ids = new List<int>();
    return dfs(root, 0, 0);
    }
    
    private List<int> ids;
  
  private int dfs(TreeNode root, int d, int id) {
    if (root == null) return 0;
    if (ids.Count == d) ids.Add(id);
    return Math.Max(id - ids[d] + 1, 
             Math.Max(this.dfs(root.left, d + 1, (id - ids[d]) * 2),
                      this.dfs(root.right, d + 1, (id - ids[d]) * 2 + 1)));
  }
}

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

// 

// 

// 

// 

//

// 

// 

// 

// 

// 

// 

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//

//



