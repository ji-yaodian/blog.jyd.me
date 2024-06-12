---
title: TypeError("'ObjectId' object is not iterable") 解决方法
date: 2024-06-05T21:43:54+08:00
tags: ["python", "fastpi", "mongodb"]
categories: ["python", "mongodb"]
draft: false
---
# 'ObjectId' object is not iterable
使用fastapi，当返回结果里面包含mongodb的id，也就是ObjectId类型的时候，就会报错： TypeError("'ObjectId' object is not iterable")。

直接google搜索下，可以得到几种方法：

1. 先使用str()方法将ObjectId转换成字符串
2. 使用bson内置的json_util.dumps()方法将ObjectId转换成字符串
3. 删除ObjectId字段
4. 定义一个JSONEncoder类，将ObjectId转换成字符串

```python
import json
from bson import ObjectId

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

JSONEncoder().encode(analytics)
```

5. json.dumps(my_obj, default=str)
6. 如果是老版本的fastapi

```python
import pydantic
from bson import ObjectId
pydantic.json.ENCODERS_BY_TYPE[ObjectId]=str
```

看起来，第6种方法是比较优雅的，但是，对于没有使用pydantic的返回结果，就不适用了。而且，新版本的pydantic也不是这样的使用方法了。

其实，不管什么类型，只要是json不支持的，都会报错，比如datetime类型也会报错。
但是为什么fastapi返回datetime类型的时候不会报错呢？因为fastapi内部已经做了处理，将datetime类型转换成了字符串类型。

通过报错信息。

```shell
│ /xxxx/site-packages/fastapi/encoders.py:332 in          │
│ jsonable_encoder                                                                                 │
│                                                                                                  │
│   329 │   │   │   data = vars(obj)                                                               │
│   330 │   │   except Exception as e:                                                             │
│   331 │   │   │   errors.append(e)                                                               │
│ ❱ 332 │   │   │   raise ValueError(errors) from e                                                │
│   333 │   return jsonable_encoder(                                                               │
│   334 │   │   data,
```

我们可以看到是在fastapi的encoders.py里面报错的。

打开encoders.py文件, 可以看到

```python
    for encoder, classes_tuple in encoders_by_class_tuples.items():
        if isinstance(obj, classes_tuple):
            return encoder(obj)
```

有点眼熟是不是，跟JSONEncoder类似，只不过是fastapi内部的实现。
稍微看一眼代码，可以看到：

```python

ENCODERS_BY_TYPE: Dict[Type[Any], Callable[[Any], Any]] = {
    bytes: lambda o: o.decode(),
    Color: str,
    datetime.date: isoformat,
    datetime.datetime: isoformat,
    datetime.time: isoformat,
    datetime.timedelta: lambda td: td.total_seconds(),
    Decimal: decimal_encoder,
    Enum: lambda o: o.value,
    frozenset: list,
    deque: list,
    GeneratorType: list,
    IPv4Address: str,
    IPv4Interface: str,
    IPv4Network: str,
    IPv6Address: str,
    IPv6Interface: str,
    IPv6Network: str,
    NameEmail: str,
    Path: str,
    Pattern: lambda o: o.pattern,
    SecretBytes: str,
    SecretStr: str,
    set: list,
    UUID: str,
    Url: str,
    AnyUrl: str,
}
```

这个就是fastapi内部的处理方式，将不支持的类型通过对应的处理方式转换成支持的类型。

那么，我们就得到一种比较简单的处理方法了。在程序启动之前，将ObjectId类型添加到ENCODERS_BY_TYPE里面，调用str方法转换即可。

```python
from bson import ObjectId
from fastapi.encoders import ENCODERS_BY_TYPE

#fastapi默认不支持ObjectId转json，这里做个映射，调用str方法转换
ENCODERS_BY_TYPE[ObjectId] = str
```

这样，就可以解决fastapi返回ObjectId类型的问题了。如果有其他类型要处理，也可以按照这个方法处理。


fastapi==0.111.0
