# 6 · Merge Two Sorted Arrays

Merge two given sorted ascending integer array A and B into a new sorted integer array.

Example 1:

Input:

	A = [1]
	B = [1]
Output:

	[1,1]
Explanation:

return array merged.

Example 2:

Input:

	A = [1,2,3,4]
	B = [2,4,5,6]
Output:

	[1,2,2,3,4,4,5,6]

## 代码

	from typing import (
	    List,
	)
	
	class Solution:
	    """
	    @param a: sorted integer array A
	    @param b: sorted integer array B
	    @return: A new sorted integer array
	    """
	    def merge_sorted_array(self, a: List[int], b: List[int]) -> List[int]:
	        # write your code here
	        i, j = 0, 0
	        ans = []
	        # 如果没有走到队尾
	        while i < len(a) and j < len(b):
	            #如果队列a的数小于队列b的数
	            if a[i] < b[j]:
	                #添加a的数到返回参数
	                ans.append(a[i])
	                #队列a的指针往后移动
	                i += 1
	            else: # 如果队列a的数大于队列b的数
	                #添加b的数到返回参数
	                ans.append(b[j])
	                #队列b的指针往后移动
	                j += 1
	
	        # 如果b队列走完了
	        while i < len(a):
	            ans.append(a[i])
	            i += 1
	
	        # 如果a队列走完了
	        while j < len(b):
	            ans.append(b[j])
	            j += 1
	            
	        return ans