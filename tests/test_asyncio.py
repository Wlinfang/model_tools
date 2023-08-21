import asyncio
import time

async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)


async def main():
    print(f"started at {time.strftime('%X')}")
    await say_after(1, 'hello')
    await say_after(2, 'world')
    print(f"finished at {time.strftime('%X')}")

async def main2():
    print(f"started at {time.strftime('%X')}")
    task1=asyncio.create_task(say_after(1,'hello2'))
    task2=asyncio.create_task(say_after(2,'world2'))
    print(f"task created at {time.strftime('%X')}")
    await task1
    await task2
    print(f"finished at {time.strftime('%X')}")

def test_1():
    asyncio.run(main())
    asyncio.run(main2())

async def nested():
    return 42
async def main3():
    task=asyncio.create_task(nested())
    print(await task)
def test_2():
    asyncio.run(main3())