# coding=utf-8
import threading
import time
lock = threading.Lock()

def chiHuoGuo(people, do):
    with lock:
        print("%s 吃火锅的小伙伴：%s" % (time.ctime(),people))
        time.sleep(1)
        for i in range(3):
            time.sleep(1)
            print("%s %s正在 %s 鱼丸"% (time.ctime(), people, do))
        print("%s 吃火锅的小伙伴：%s" % (time.ctime(),people))


class myThread (threading.Thread):   # 继承父类threading.Thread
      # 线程锁

    def __init__(self, people, name, do):
        '''重写threading.Thread初始化内容'''
        threading.Thread.__init__(self)
        self.threadName = name
        self.people = people
        self.do = do

    def run(self):   # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        '''重写run方法'''
        print("开始线程: " + self.threadName)

        # 执行任务之前锁定线程
        # self.lock.acquire()
        with lock:
            chiHuoGuo(self.people, self.do)     # 执行任务

        # 执行完之后，释放锁
        # self.lock.release()

        print("qq交流群：226296743")
        print("结束线程: " + self.name)

print("yoyo请小伙伴开始吃火锅：！！！")

# 设置线程组
threads = []

# 创建新线程
thread1 = myThread("xiaoming", "Thread-1", "添加")
thread2 = myThread("xiaowang", "Thread-2", "吃掉")

# 添加到线程组
threads.append(thread1)
threads.append(thread2)

# 开启线程
for thread in threads:
    thread.start()

# 阻塞主线程，等子线程结束
for thread in threads:
    thread.join()

time.sleep(0.1)
print("退出主线程：吃火锅结束，结账走人")