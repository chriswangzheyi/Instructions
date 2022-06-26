# 1141 · The month's days

## 题目

Given a year and month, return the days of that month.
The conditions for a leap year are

a year divisible by 4 but not divisible by 100
a year divisible by 400.
A leap year is one of the above two conditions.

## 代码

	class Solution:
	    """
	    @param year: a number year
	    @param month: a number month
	    @return: return the number of days of the month.
	    """
	    def get_the_month_days(self, year: int, month: int) -> int:
	        # write your code here
	        if month ==2:
	            if year % 4 == 0 and year % 100 !=0:
	                return 29
	            else:
	                return 28
	        elif month==1 or month ==3 or month==5 or month==7 or month==8 or month==10 or month ==12:
	            return 31
	        else:
	            return 30
