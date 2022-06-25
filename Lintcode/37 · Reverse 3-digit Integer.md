# 37 · Reverse 3-digit Integer

Reverse a 3-digit integer.

Example 1:

Input:

	number = 123
Output:

	321
Explanation:

Reverse the number.

Example 2:

Input:

	number = 900
Output:

	9
Explanation:

Reverse the number.


## 代码

	class Solution:
	    """
	    @param number: A 3-digit number.
	    @return: Reversed number.
	    """
	    def reverse_integer(self, number: int) -> int:
	        # write your code here
	        return int(str(number)[::-1])