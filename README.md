# Wargaming CSSIM 算法仓库

本仓库以 `PythonCode` 为算法和 Python 框架的源代码。CSSIM 客户端和服务端目录是部署副本，不直接作为开发目录。

## 工作流程

1. 在 CSSIM 平台创建或下载新算法后，先执行：

   ```powershell
   .\tools\sync_from_cssim.ps1
   ```

2. 检查新增的 `PythonCode\TeamAlg\<队伍ID>`，提交一个 Git 版本。
3. 以后只在本仓库的 `PythonCode\TeamAlg\<队伍ID>` 中修改算法。
4. 运行验证：

   ```powershell
   .\tools\verify.ps1
   ```

5. 提交并推送到 GitHub。
6. 将仓库代码覆盖到 CSSIM 客户端和服务端：

   ```powershell
   .\tools\sync_to_cssim.ps1
   ```

7. 再启动 CSSIM 进行推演。

也可以用一个命令完成验证、提交、推送和同步：

```powershell
.\tools\publish.ps1 -Message "describe your change"
```

不要直接在 `E:\CSSIM\Client\Data\AlgData\PythonCode` 或 `Server\Data\AlgData\PythonCode` 中长期修改；平台新建算法后，先用 `sync_from_cssim.ps1` 把真实队伍目录导入仓库。
