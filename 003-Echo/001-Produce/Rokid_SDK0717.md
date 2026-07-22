# Rokid CXR SDK 接入笔记

> 基于 Rokid 官方 Android SDK 文档整理,对照 Echo 项目代码验证.

## 概述

Rokid 眼镜与手机通信分两层 SDK:

| SDK | 端 | 构件 |
|-----|-----|------|
| CXR-L | 手机端 | `com.rokid.cxr:client-l:1.0.4` |
| CXR-S | 眼镜端 | `com.rokid.cxr:cxr-service-bridge` |

手机端称为 Link 角色(主导鉴权/建会话),眼镜端称为 CustomApp 角色(被动显示状态+上报按键).

## 三端通信全景

① **开机** — 眼镜启动App并subscribe,手机CXRLink鉴权后appStart拉起眼镜.
② **选场景** — 镜片显示Onsite/Meeting/Quality Time三个选项.
③ **单击镜腿开始** — 眼镜sendMessage(START+scene) → 手机handleStart() → 创建后台session.
④ **录制中** — 镜片显示"记忆中",其他场景置灰.手机端RecordingController处理,固件传音频/照片.
⑤ **再单击结束** — 眼镜sendMessage(STOP) → 手机handleStop() → POST /complete,后台开始处理.
⑥ **回执** — 手机通过rk_custom_client回传处理进度,眼镜显示"上传完成".

## 关键接口

### 手机端(CXR-L)

```kotlin
// 1. 鉴权 — 依赖 Rokid AI App
AuthorizationHelper(this).authorize(requestCode)  // 唤起 AI App 授权页

// 2. 建会话 — 拉起眼镜端 App
CXRLink.createSession(
    config = CustomAppConfig(
        packageName = "com.echo.glasses",  // 须与眼镜端 applicationId 一致
        scene = "ONSITE"
    )
).appStart()  // 眼镜端收到 → 启动 MainActivity

// 3. 接收眼镜上报的按键
glassesConnection.commands.collect { command ->
    when (command.type) {
        START -> handleStart(command)  // 开始录制
        STOP  -> handleStop()          // 结束录制
    }
}
```

### 眼镜端(CXR-S)

```kotlin
// 1. 初始化 — appStart 被拉起后尽早完成
CXRServiceBridge().apply {
    setStatusListener(object : StatusListener {
        onConnected     → 镜片显示"手机已连接"
        onDisconnected  → 镜片显示"手机已断开"
    })
    subscribe("rk_custom_client", msgCallback)  // 接收手机下发的自定义指令
}

// 2. 上报按键
bridge.sendMessage(
    "rk_custom_key",
    Caps().apply {
        write("cmd"); write("START")
        write(scene.cmd)  // "ONSITE" / "MEETING" / "QUALITY_TIME"
    }
)

// 3. 按键类型
KeyType.CLICK                    → 单击: START / STOP
KeyType.TWO_FINGER_SWIPE_FORWARD → 前滑: 切换场景
KeyType.TWO_FINGER_SWIPE_BACK    → 后滑: 切换场景
```

## 已发现的问题

### 1. sendMessage 返回值未检查

眼镜端 `startMemory()` 调了 `bridge.sendMessage(...)`,但未检查返回值 `ret`.CXR 断开时 `ret` 为负值,但 `recording = true` 仍执行,镜片显示"记忆中"但手机实际没收到.

**修复:** 检查返回值,发失败时不下发 recording 状态,提示用户重试.

### 2. LocalBlobStore 接口名不匹配

手机端视频上传后调用 `blob.put()`,但 `LocalBlobStore` 只有 `save()`.

**修复:** 统一接口名(psh 已知,待修).

### 3. 手机端断联无提示

眼镜端有 `onDisconnected → "手机已断开"`,但手机端没有任何断联提示.

**优化方向:** 手机端增加断联弹窗/震动提示.

## 配置要点

### Maven 仓库

```kotlin
// settings.gradle.kts
maven { url = uri("https://maven.rokid.com/repository/maven-public/") }
```

### minSdk
```kotlin
// app/build.gradle.kts
minSdk = 31  // Android 12+
```

### 权限(手机端)
```xml
<!-- AndroidManifest.xml 必需 -->
<uses-permission android:name="android.permission.INTERNET"/>
<!-- CustomApp 安装 APK 时需要 -->
<uses-permission android:name="android.permission.MANAGE_EXTERNAL_STORAGE"/>
```

## 补充说明

1. **FileProvider 配置** — 分享本地音频WAV文件时需要.当前音频导出分享功能未做,故暂不写入.
2. **appUploadAndInstall 存储权限校验** — 若眼镜端APK需从手机端安装时使用.现眼镜App已预装,故跳过.
