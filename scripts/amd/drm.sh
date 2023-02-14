#!/bin/bash
# From https://github.com/pytorch/builder/blob/main/common/install_rocm_drm.sh

###########################
### prereqs
###########################
# Install Python packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    apt-get update -y
    apt-get install -y libpciaccess-dev pkg-config
    apt-get clean
    ;;
  centos)
    yum install -y libpciaccess-devel pkgconfig
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
python3 -m pip install meson ninja


###########################
### clone repo
###########################
GIT_SSL_NO_VERIFY=true git clone https://gitlab.freedesktop.org/mesa/drm.git
pushd drm

###########################
### patch
###########################
patch -p1 <<'EOF'
diff --git a/amdgpu/amdgpu_asic_id.c b/amdgpu/amdgpu_asic_id.c
index a5007ffc..8ae66f28 100644
--- a/amdgpu/amdgpu_asic_id.c
+++ b/amdgpu/amdgpu_asic_id.c
@@ -22,6 +22,13 @@
  *
  */

+#define _XOPEN_SOURCE 700
+#define _LARGEFILE64_SOURCE
+#define _FILE_OFFSET_BITS 64
+#include <ftw.h>
+#include <link.h>
+#include <limits.h>
+
 #include <ctype.h>
 #include <stdio.h>
 #include <stdlib.h>
@@ -29,11 +36,27 @@
 #include <string.h>
 #include <unistd.h>
 #include <errno.h>
+#include <dlfcn.h>

 #include "xf86drm.h"
 #include "amdgpu_drm.h"
 #include "amdgpu_internal.h"

+static char *amdgpuids_path = NULL;
+
+static int check_for_location_of_amdgpuids(const char *filepath, const struct stat *info, const int typeflag, struct FTW *pathinfo)
+{
+	if (typeflag == FTW_F && strstr(filepath, "amdgpu.ids")) {
+		if (NULL != amdgpuids_path) {
+			free(amdgpuids_path);
+		}
+		amdgpuids_path = strdup(filepath);
+		return 0;
+	}
+
+	return 0;
+}
+
 static int parse_one_line(struct amdgpu_device *dev, const char *line)
 {
	char *buf, *saveptr;
@@ -113,6 +135,33 @@ void amdgpu_parse_asic_ids(struct amdgpu_device *dev)
	int line_num = 1;
	int r = 0;

+	char self_path[ PATH_MAX ];
+	ssize_t count;
+	ssize_t rc;
+	ssize_t i;
+
+	fp = NULL;
+	Dl_info info;
+
+	rc = dladdr( "amdgpu_parse_asic_ids", &info );
+	count = strlen(info.dli_fname);
+	strcpy(self_path, info.dli_fname);
+	for (i=count; i>0; --i) {
+		if (self_path[i] == '/') break;
+			self_path[i] = '\0';
+		}
+
+		if (0 == nftw(self_path, check_for_location_of_amdgpuids, 5, FTW_PHYS)) {
+			if (amdgpuids_path) {
+				fp = fopen(amdgpuids_path, "r");
+				if (!fp) {
+					fprintf(stderr, "%s: %s\n", amdgpuids_path, strerror(errno));
+				}
+			}
+		}
+
+	if (!fp) {
+
	fp = fopen(AMDGPU_ASIC_ID_TABLE, "r");
	if (!fp) {
		fprintf(stderr, "%s: %s\n", AMDGPU_ASIC_ID_TABLE,
@@ -120,6 +171,8 @@ void amdgpu_parse_asic_ids(struct amdgpu_device *dev)
		return;
	}

+ }
+
	/* 1st valid line is file version */
	while ((n = getline(&line, &len, fp)) != -1) {
		/* trim trailing newline */
diff --git a/amdgpu/meson.build b/amdgpu/meson.build
index 20bf6244..700ab38a 100644
--- a/amdgpu/meson.build
+++ b/amdgpu/meson.build
@@ -36,7 +36,7 @@ libdrm_amdgpu = library(
   ],
   include_directories : [inc_root, inc_drm],
   link_with : libdrm,
-  dependencies : [dep_threads, dep_atomic_ops, dep_rt],
+  dependencies : [dep_threads, dep_atomic_ops, dep_rt, dep_dl],
   version : '1.0.0',
   install : true,
 )
diff --git a/meson.build b/meson.build
index 3c2e171d..7929dd12 100644
--- a/meson.build
+++ b/meson.build
@@ -190,11 +190,8 @@ endif
 summary('VC4', with_vc4)

 # Among others FreeBSD does not have a separate dl library.
-if not cc.has_function('dlsym')
-  dep_dl = cc.find_library('dl', required : with_nouveau)
-else
-  dep_dl = []
-endif
+dep_dl = cc.find_library('dl', required : with_amdgpu)
+
 # clock_gettime might require -rt, or it might not. find out
 if not cc.has_function('clock_gettime', prefix : '#define _GNU_SOURCE\n#include <time.h>')
   # XXX: untested
--
EOF

###########################
### build
###########################
meson builddir --prefix=$PWD
pushd builddir
ninja install
popd
popd

if [[ -n "${TRITON_ROCM_DIR}" ]]; then
  echo "Copying DRM Libraries"
  mkdir -p $TRITON_ROCM_DIR/lib/data
  cp drm/builddir/amdgpu/libdrm_amdgpu.so.1 $TRITON_ROCM_DIR/lib/
  cp drm/builddir/libdrm.so.2 $TRITON_ROCM_DIR/lib/
  cp drm/data/amdgpu.ids $TRITON_ROCM_DIR/lib/data/
fi

rm -rf drm
