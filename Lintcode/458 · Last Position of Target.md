# 458 · Last Position of Target

## 题目

Find the last position of a target number in a sorted array. Return -1 if target does not exist.

Example 1:

	Input: nums = [1,2,2,4,5,5], target = 2
	Output: 2
	
Example 2:

	Input: nums = [1,2,2,4,5,5], target = 6
	Output: -1


##  代码

	from typing import (
	    List,
	)
	
	class Solution:
	    """
	    @param nums: An integer array sorted in ascending order
	    @param target: An integer
	    @return: An integer
	    """
	    def last_position(self, nums: List[int], target: int) -> int:
	        # write your code here
	        if not nums:
	            return -1
	
	        left = 0 
	        right = len(nums) - 1
	
		 # 关键步骤：考虑到相同指，取右边的数，则需要left + 1 
	        while left + 1 < right:
	            mid = left + (right - left) // 2
	            if nums[mid] > target:
	                right = mid
	            else:
	                left = mid
	
	        if nums[right] == target:
	            return right
	        if nums[left] == target:
	            return left
	        return -1