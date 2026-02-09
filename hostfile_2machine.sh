for i in {0..1}; do
  var_name="GEMINI_IP_taskrole1_$i"
  echo "${!var_name} slots=8"
done > ./hostfile_2machine 
