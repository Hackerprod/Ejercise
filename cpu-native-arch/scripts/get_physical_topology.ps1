$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class PhysicalTopology {
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetLogicalProcessorInformationEx(
        int relationship, IntPtr buffer, ref uint returnedLength);

    public static void Dump() {
        uint length = 0;
        GetLogicalProcessorInformationEx(0, IntPtr.Zero, ref length);
        IntPtr buffer = Marshal.AllocHGlobal((int)length);
        try {
            if (!GetLogicalProcessorInformationEx(0, buffer, ref length)) {
                throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
            }
            long offset = 0;
            int core = 0;
            while (offset < length) {
                IntPtr record = IntPtr.Add(buffer, (int)offset);
                uint size = (uint)Marshal.ReadInt32(record, 4);
                ushort groupCount = (ushort)Marshal.ReadInt16(record, 30);
                Console.WriteLine("core=" + core + " group_count=" + groupCount);
                for (int groupIndex = 0; groupIndex < groupCount; groupIndex++) {
                    IntPtr groupMask = IntPtr.Add(record, 32 + groupIndex * 32);
                    ulong mask = (ulong)Marshal.ReadInt64(groupMask);
                    ushort group = (ushort)Marshal.ReadInt16(groupMask, 8);
                    Console.WriteLine("  group=" + group + " mask=0x" + mask.ToString("X16"));
                    for (int bit = 0; bit < 64; bit++) {
                        if ((mask & (1UL << bit)) != 0) {
                            Console.WriteLine("  logical_cpu=" + bit + " group=" + group);
                        }
                    }
                }
                offset += size;
                core++;
            }
        }
        finally {
            Marshal.FreeHGlobal(buffer);
        }
    }
}
'@

[PhysicalTopology]::Dump()
