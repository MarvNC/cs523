For fixing markdown headers:

```regex
(^#+)(?=[^ #])
```

Replace with `$1 ` in VSCode advanced search with only markdown source enabled
