---
title: TypeError("'ObjectId' object is not iterable") 
date: 2024-06-05T21:43:54+08:00
tags: ["python", "fastpi", "mongodb"]
categories: ["python", "mongodb"]
draft: false
---
# 'ObjectId' object is not iterable
If you use fastapi, when the return result contains the id of mongodb, which is the ObjectId type, the error message will be reported: TypeError("'ObjectId' object is not iterable").

A direct Google search can be used to get several methods:

1. Use the str() method to convert the ObjectId to a string
2. Use the built-in json_util.dumps() method of bson to convert the ObjectId to a string
3. Delete the ObjectId field
4. Define a JSONEncoder class to convert the ObjectId to a string

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
6. If it is an older version of fastapi

```python
import pydantic
from bson import ObjectId
pydantic.json.ENCODERS_BY_TYPE[ObjectId]=str
```

It seems that the 6th method is more elegant, but it does not work for the return results that do not use pydantic. Also, the new version of pydantic doesn't work that way.

In fact, no matter what type, as long as it is not supported by JSON, an error will be reported, such as the datetime type.
But why doesn't fastapi report an error when it returns the datetime type? This is because FastAPI has done the internal processing to convert the datetime type to the string type.

Through the error message.

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

We can see that the error is reported in the fastapi encoders.py.

Open encoders.py file, you can see

```python
    for encoder, classes_tuple in encoders_by_class_tuples.items():
        if isinstance(obj, classes_tuple):
            return encoder(obj)
```

It's a bit familiar, it's similar to JSONEncoder, but it's an internal implementation of fastapi.
A little glance at the code shows that:

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

This is the internal processing method of fastapi, which converts unsupported types into supported types through the corresponding processing methods.

So, we get a relatively simple way to deal with it. Before the program starts, add the ObjectId type to the ENCODERS_BY_TYPE and call the str method to convert.


```python
from bson import ObjectId
from fastapi.encoders import ENCODERS_BY_TYPE

#fastapi does not support converting ObjectId to json by default, so make a mapping here and call the str method to convert
ENCODERS_BY_TYPE[ObjectId] = str
```

In this way, the problem of fastapi returning an ObjectId type can be solved. If there are other types to be processed, you can also do the same.

fastapi==0.111.0