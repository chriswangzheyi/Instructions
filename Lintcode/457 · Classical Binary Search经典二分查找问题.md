# 457 · Classical Binary Search

## 题目

Find any position of a target number in a sorted array. Return -1 if target does not exist.


Example 1:

	Input: nums = [1,2,2,4,5,5], target = 2
	Output: 1 or 2
Example 2:

	Input: nums = [1,2,2,4,5,5], target = 6
	Output: -1

##  代码（二分法）

	class Solution:
	    """
	    @param nums: An integer array sorted in ascending order
	    @param target: An integer
	    @return: An integer
	    """
	    def findPosition(self, nums, target):
	        # write your code here
	        low, high = 0, len(nums) - 1
	        while low <= high:
	            mid = (high - low) // 2 + low
	            num = nums[mid]
	            if num == target:
	                return mid
	            elif num > target:
	                high = mid - 1
	            else:
	                low = mid + 1
	        return -1