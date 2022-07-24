# 14 · First Position of Target


## 题目

Given a sorted array (ascending order) and a target number, find the first index of this number in O(log n)O(logn) time complexity.


Example 1:

Input:

	tuple = [1,4,4,5,7,7,8,9,9,10]
	target = 1
	
Output:

	0
Explanation:

The first index of 1 is 0.

Example 2:

Input:

	tuple = [1, 2, 3, 3, 4, 5, 10]
	target = 3
	
Output:

	2

Explanation:

The first index of 3 is 2.

Example 3:

Input:

	tuple = [1, 2, 3, 3, 4, 5, 10]
	target = 6

Output:

	-1

Explanation:

There is no 6 in the array，return -1.

## 代码 (二分法)

	from typing import (
	    List,
	)
	
	class Solution:
	    """
	    @param nums: The integer array.
	    @param target: Target to find.
	    @return: The first position of target. Position starts from 0.
	    """
	    def binary_search(self, nums: List[int], target: int) -> int:
	        # write your code here
	        if not nums:
	            return -1
	
	        low, high = 0, len(nums) - 1
	        while low <= high:
	            mid = (high + low) // 2
	            if nums[mid] >= target:
	                high = mid - 1
	            else:
	                low = mid + 1
	        if nums[low] == target: return low
	        return -1
	        
## 特别注意的点

	while low <= high:
	      
这一行需要确保有“等于”  