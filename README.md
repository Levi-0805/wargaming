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

   部署脚本要求当前在干净的 `main` 分支，且 `origin/main` 已与本地提交一致；部署前会把客户端和服务端的算法库备份到 `E:\CSSIM\Backups\<时间戳>`，部署后逐文件校验 SHA-256。目标目录中与仓库无关的额外文件会保留。

7. 再启动 CSSIM 进行推演。

也可以用一个命令完成验证、提交、推送和同步：

```powershell
.\tools\publish.ps1 -Message "describe your change"
```

仓库中的 `PythonCode\TeamAlg\No1RuleRed` 是一个可运行的 `No.1` 红方规则算法示例。使用
`PythonCode\TaskConfig\cssim\no1_rule_red.jsonc` 可让红方加载它、蓝方加载官方默认规则。

不要直接在 `E:\CSSIM\Client\Data\AlgData\PythonCode` 或 `Server\Data\AlgData\PythonCode` 中长期修改；平台新建算法后，先用 `sync_from_cssim.ps1` 把真实队伍目录导入仓库。
