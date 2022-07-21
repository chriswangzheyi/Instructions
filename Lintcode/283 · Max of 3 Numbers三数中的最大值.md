# 283 · Max of 3 Numbers

## 题目

Example 1:

	Input:  num1 = 1, num2 = 9, num3 = 0
	Output: 9
	
	Explanation: 
	return the Max of them.

Example 2:

	Input:  num1 = 1, num2 = 2, num3 = 3
	Output: 3
	
	Explanation: 
	return the Max of them.
	
## 代码

	class Solution:
	    """
	    @param num1: An integer
	    @param num2: An integer
	    @param num3: An integer
	    @return: an interger
	    """
	    def max_of_three_numbers(self, num1: int, num2: int, num3: int) -> int:
	        # write your code here
	        if num1 < num3:
	            return max(num2,num3)
	        else:
	            return max (num1,num2)