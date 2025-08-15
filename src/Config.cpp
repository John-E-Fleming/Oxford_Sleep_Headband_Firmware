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
        } else if (key == "sample_rate") {
          cfg.sample_rate = val.toInt();
        } else if (key == "channels") {
          cfg.channels = val.toInt();
        } else if (key == "format") {
          cfg.format = val;
        } else if (key == "gain") {
          cfg.gain = val.toInt();
        } else if (key == "vref") {
          cfg.vref = val.toFloat();
        } else if (key == "bipolar_channel_positive") {
          cfg.bipolar_channel_positive = val.toInt();
        } else if (key == "bipolar_channel_negative") {
          cfg.bipolar_channel_negative = val.toInt();
        } else if (key == "ml_target_sample_rate") {
          cfg.ml_target_sample_rate = val.toInt();
        } else if (key == "ml_window_seconds") {
          cfg.ml_window_seconds = val.toInt();
        }
      }
      line = "";
    } else {
      line += c;
    }
  }
  cfgFile.close();
  return cfg.datafile.length() > 0;
}
