###############
# @author Jeff
# @date 2018-09-28
#######################

__all__ = ["helloAgain"]


from testpackage.test1 import hello1
from testpackage.test2 import hello2

def helloAgain(name):
    hello1(name)
    hello2(name)




