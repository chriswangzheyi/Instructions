# 1910 · Most numbers

## 

Find the number with the most occurrences in the given array.
When the number of occurrences is the same, return the smallest one.

Example 1:

Input: 

	[1,1,2,3,3,3,4,5]
	
Output: 

	3
Example 2:

Input: 

	[1]
	
Output: 

	1

## 代码

	from typing import (
	    List,
	)
	
	class Solution:
	    """
	    @param array: An array.
	    @return: An interger.
	    """
	    def find_number(self, array: List[int]) -> int:
	        # Write your code here.
	        max = 0
	        res = array[0]
	        dict1 = dict()
	        for c in array:
	            if  c not in dict1:
	                dict1[c] = 1
	            else:
	                dict1[c] += 1
	            
	            if dict1[c] > max or (dict1[c] == max and c < res):
	                max = dict1[c]
	                res = c
	        return res