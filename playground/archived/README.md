# 已封存的研究

本目录保存停止维护的实验代码与历史判断，不作为当前研究的依赖、运行入口或验收基准。

| 工作区                     | 封存日期   | 状态                                                                |
| -------------------------- | ---------- | ------------------------------------------------------------------- |
| `adaptive_bci_sampling/`   | 2026-10-18 | 全场误差与参数采样路线封存，不再启用                                |
| `dtdm/`                    | 2026-10-20 | ASP-DAC 2026 DTDM 复现封存，不再维护                                |
| `hdim/`                    | 2026-10-20 | ISEDA 2025 层级分解复现封存，不再维护                               |
| `nonlinear_interface_rom/` | 2026-10-26 | 瞬态非线性界面 ROM 沙箱封存，结论保留、代码不再维护                 |
| `cms_prototype/`           | 2026-10-28 | 带空腔封装的部件 ROM + TIM/lid FOM 原型封存，结论保留、代码不再维护 |

`adaptive_bci_sampling/` 从提交 `c776a7552366b322ad1cd043dff622c7e449dba6` 原样迁入。内部 README 的“维护中”状态、运行命令和旧路径属于历史记录，不代表当前维护承诺；迁移后不修补其运行环境或路径。

`dtdm/` 与 `hdim/` 从 `playground/` 根目录原样迁入，并已从 `playground/CMakeLists.txt` 移除 `add_subdirectory`，构建系统中不再存在 `mhs_dtdm`、`mhs_hdim` 目标。两者 CMakeLists 与 README 中残留的构建命令属于历史记录。

`nonlinear_interface_rom/` 从 `playground/` 根目录原样迁入，其 README 记录的结论与撤回保留了完整推导与证据等级，作为历史判断参考。该目录从未接入构建系统，迁移不涉及 CMake 或依赖变更；内部运行命令与 `../../python` 相对路径按迁移后的位置已失效，属于历史记录，不修补。

`cms_prototype/` 从 `playground/` 根目录原样迁入。它从未接入 `playground/CMakeLists.txt`，迁移不涉及 CMake 或依赖变更；内部 README 中的 `playground/cms_prototype/...` 运行命令与输出路径按迁移后的位置已失效，2026-10-27 的核验记录作为历史证据保留，不重跑、不修补。

`structured_interface_operator/` 于 2026-10-26 整目录删除，未迁入本目录。该工作区的 Robin HTC 路线停在“连续 HTC × 全频率无证书”：脚本是可运行的数值原型，核心问题（不保留边界未知量、不依赖参数多项式 BRL 的连续 HTC 上界）仍无证明或有效实例，因此不保留归档副本。它引用的材料参数盒归档 `structured_interface_operator_material_20261018/` 在任何提交中都不存在，是原记录遗留的失效链接。

后续研究在独立工作区开展，不导入或重新启用本目录下的脚本。
