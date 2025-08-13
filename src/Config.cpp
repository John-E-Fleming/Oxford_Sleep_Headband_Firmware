#include "Config.h"

bool loadConfig(SdFile &cardRoot, Config &cfg) {
  SdFile cfgFile;
  if (!cfgFile.open("config.txt", O_READ)) return false;

  String line;
  while (cfgFile.available()) {
    char c = cfgFile.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line.trim();
      int eq = line.indexOf('=');
      if (eq > 0) {
        String key = line.substring(0, eq);
        String val = line.substring(eq + 1);
        key.trim(); val.trim();
        if (key == "datafile") {
          cfg.datafile = val;
        }
        // else if (key == "mode") { … }
      }
      line = "";
    } else {
      line += c;
    }
  }
  cfgFile.close();
  return cfg.datafile.length() > 0;
}
